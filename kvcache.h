#pragma once
// kvcache.h — KV-Cache + Autoregressive decoding.
// ===========================================================================
// KV Caching:
// ===========================================================================
// K and V vectors for past tokens DO NOT CHANGE between decode
// steps. Only Q changes (as we only care about the new token's query).
// So we cache K and V, and for each new token only compute:
//   - Q for the new token
//   - K, V for the new token (append to cache)
//   - attention scores between the new Q and ALL cached K/V
//
// This reduces the per-step cost from O(N²) -> O(N) in total.
//
// Memory cost: 2 (K and V) × n_layers × max_seq × d_model × 4 bytes
//   For GPT-2 small: 2 × 12 × 1024 × 768 × 4 = ~75 MB
//   Pre-allocated once at startup. Zero heap traffic on the decode path.
//
// ===========================================================================
// Two-phase decoding:
// ===========================================================================
//   Prefill: run the full forward pass on the prompt (seq_len > 1).
//            Populate the KV-cache with K/V for all prompt positions.
//            Return logits for the LAST prompt position only (next token).
//
//   Decode: for each new token:
//            1. Compute Q, K, V for this single token.
//            2. Append K, V to the cache.
//            3. Compute attention scores between this Q and ALL cached K/V.
//            4. Run FFN, residual, etc.
//            5. Return logits. Pick next token. Repeat.

#include "gpt2.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

// ===========================================================================
// LayerKVCache — stores K and V vectors for one transformer block.
// ===========================================================================
// K_cache shape: {max_seq, d_model}  — one row per cached token
// V_cache shape: {max_seq, d_model}
// n_cached: how many positions have been filled so far
struct LayerKVCache {
    Tensor K_cache;  // {max_seq, d_model}
    Tensor V_cache;  // {max_seq, d_model}
    int    n_cached = 0;

    LayerKVCache() = default;

    explicit LayerKVCache(const GPT2Config& cfg)
        : K_cache({cfg.max_seq, cfg.d_model}),
          V_cache({cfg.max_seq, cfg.d_model}),
          n_cached(0) {}

    // Append K and V vectors for a new token position.
    // kv has shape {1, 3*d_model}; K is at offset d_model, V at 2*d_model.
    void append(const float* k_vec, const float* v_vec, int d_model) {
        assert(n_cached < K_cache.shape[0]);
        float* k_dst = K_cache.data.data() + n_cached * d_model;
        float* v_dst = V_cache.data.data() + n_cached * d_model;
        std::memcpy(k_dst, k_vec, d_model * sizeof(float));
        std::memcpy(v_dst, v_vec, d_model * sizeof(float));
        ++n_cached;
    }
};

// One KV-cache per transformer block.
struct KVCache {
    std::vector<LayerKVCache> layers;

    KVCache() = default;

    explicit KVCache(const GPT2Config& cfg) {
        layers.reserve(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; ++i)
            layers.emplace_back(cfg);
    }

    void reset() {
        for (auto& l : layers) l.n_cached = 0;
    }
};


// ===========================================================================
// attention_cached() — single-token attention using the KV-cache.
// ===========================================================================
// INPUT:  x_t   {d_model}   — the single new token's hidden state
//         bw                — block weights
//         cache             — K/V cache for this layer (will be updated)
//         cfg
// OUTPUT: {d_model}         — attended output for this token
//
// This is called during the DECODE phase (one token at a time).
// During PREFILL we use the original attention() which handles seq_len > 1.
inline Tensor attention_cached(const Tensor& x_t,
                                const BlockWeights& bw,
                                LayerKVCache& cache,
                                const GPT2Config& cfg) {
    int d_model = cfg.d_model;
    int n_heads = cfg.n_heads;
    int d_head  = cfg.d_head;

    // -----------------------------------------------------------------------
    // Step 1: Compute Q, K, V for this single token
    // -----------------------------------------------------------------------
    Tensor qkv = matvec(bw.c_attn_w, x_t);   // {3*d_model}
    add_bias(qkv, bw.c_attn_b);

    // Split into Q, K, V slices of length d_model each
    const float* q_ptr = qkv.data.data();
    const float* k_ptr = qkv.data.data() + d_model;
    const float* v_ptr = qkv.data.data() + 2 * d_model;

    // -----------------------------------------------------------------------
    // Step 2: Append K, V to the cache
    // -----------------------------------------------------------------------
    cache.append(k_ptr, v_ptr, d_model);
    int n_ctx = cache.n_cached;  // total context length including this token

    // -----------------------------------------------------------------------
    // Step 3: Multi-head attention — this token attends to ALL cached positions
    // -----------------------------------------------------------------------
    // Q from this token: shape {n_heads, d_head}
    // K_cache[0..n_ctx-1]: each row is {d_model} = {n_heads * d_head}
    //
    // For each head h:
    //   q_h = q_ptr + h*d_head              (length d_head)
    //   k_h[j] = K_cache[j] + h*d_head      (length d_head, for j=0..n_ctx-1)
    //   score[j] = dot(q_h, k_h[j]) / sqrt(d_head)
    //   weights = softmax(score)             (length n_ctx — no mask needed!
    //                                         we only have one Q, and it can
    //                                         attend to all past K positions)
    //   out_h = sum_j(weights[j] * v_h[j])  (length d_head)

    float scale = 1.0f / std::sqrt((float)d_head);
    Tensor attn_out({d_model});  // output for this token, all heads concatenated

    for (int h = 0; h < n_heads; ++h) {
        int head_off = h * d_head;

        const float* q_h = q_ptr + head_off;

        // Compute attention scores for all cached positions
        // scores[j] = dot(q_h, K_cache[j][head_off..head_off+d_head]) * scale
        std::vector<float> scores(n_ctx);
        for (int j = 0; j < n_ctx; ++j) {
            const float* k_j = cache.K_cache.data.data() + j * d_model + head_off;
            float dot = 0.0f;
            for (int k = 0; k < d_head; ++k)
                dot += q_h[k] * k_j[k];
            scores[j] = dot * scale;
        }

        // Softmax over scores (no causal mask needed: only one query, all
        // positions it attends to are already in the past)
        Tensor scores_t(std::vector<float>(scores.begin(), scores.end()), {n_ctx});
        softmax(scores_t);

        // Weighted sum of V values
        float* out_h = attn_out.data.data() + head_off;
        for (int k = 0; k < d_head; ++k) out_h[k] = 0.0f;
        for (int j = 0; j < n_ctx; ++j) {
            const float* v_j = cache.V_cache.data.data() + j * d_model + head_off;
            float w = scores_t.data[j];
            for (int k = 0; k < d_head; ++k)
                out_h[k] += w * v_j[k];
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Output projection
    // -----------------------------------------------------------------------
    Tensor out = matvec(bw.c_proj_w, attn_out);
    add_bias(out, bw.c_proj_b);
    return out;
}


// ===========================================================================
// transformer_block_cached() — one block, single-token, using KV-cache.
// ===========================================================================
inline Tensor transformer_block_cached(const Tensor& x_t,
                                        const BlockWeights& bw,
                                        LayerKVCache& cache,
                                        const GPT2Config& cfg) {
    // Sub-layer 1: Pre-LN + Attention + Residual
    Tensor ln1_out = x_t;
    layernorm(ln1_out, bw.ln1_gamma, bw.ln1_beta);

    Tensor attn_out = attention_cached(ln1_out, bw, cache, cfg);

    // Residual
    Tensor x2({cfg.d_model});
    for (int i = 0; i < cfg.d_model; ++i)
        x2.data[i] = x_t.data[i] + attn_out.data[i];

    // Sub-layer 2: Pre-LN + FFN + Residual
    Tensor ln2_out = x2;
    layernorm(ln2_out, bw.ln2_gamma, bw.ln2_beta);

    // FFN (operates on a single token vector)
    Tensor h = matvec(bw.mlp_fc_w, ln2_out);
    add_bias(h, bw.mlp_fc_b);
    gelu(h);
    Tensor ffn_out = matvec(bw.mlp_proj_w, h);
    add_bias(ffn_out, bw.mlp_proj_b);

    // Residual
    Tensor x3({cfg.d_model});
    for (int i = 0; i < cfg.d_model; ++i)
        x3.data[i] = x2.data[i] + ffn_out.data[i];

    return x3;
}


// ===========================================================================
// prefill() — run the full forward pass on the prompt, populate KV-cache.
// ===========================================================================
// Returns the logits for the LAST token (the prediction for the next token).
// Also populates cache.layers[i] with K/V for every prompt position.
//
// NOTE: (A more optimised impl would fuse this with gpt2_forward; here I keep
// them separate for clarity.)
inline Tensor prefill(const std::vector<int>& token_ids,
                       const GPT2Weights& weights,
                       const GPT2Config& cfg,
                       KVCache& cache) {
    cache.reset();
    int seq_len = (int)token_ids.size();
    assert(seq_len > 0 && seq_len <= cfg.max_seq);

    // Build initial embeddings: x[t] = wte[tok] + wpe[t]
    Tensor x({seq_len, cfg.d_model});
    for (int t = 0; t < seq_len; ++t) {
        int tok = token_ids[t];
        for (int d = 0; d < cfg.d_model; ++d)
            x.data[t * cfg.d_model + d] =
                weights.wte.data[tok * cfg.d_model + d] +
                weights.wpe.data[t   * cfg.d_model + d];
    }

    // Pass through all transformer blocks, populating the KV-cache at each layer
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const BlockWeights& bw = weights.blocks[layer];
        LayerKVCache& lc = cache.layers[layer];

        // LayerNorm all token vectors, then compute QKV to populate cache
        Tensor ln1_out({seq_len, cfg.d_model});
        for (int t = 0; t < seq_len; ++t) {
            Tensor xt = row(x, t);
            layernorm(xt, bw.ln1_gamma, bw.ln1_beta);
            set_row(ln1_out, t, xt);
        }

        // Compute combined QKV for all positions and populate the cache
        for (int t = 0; t < seq_len; ++t) {
            Tensor xt = row(ln1_out, t);
            Tensor qkv = matvec(bw.c_attn_w, xt);
            add_bias(qkv, bw.c_attn_b);
            // K is at offset d_model, V at 2*d_model
            const float* k_ptr = qkv.data.data() + cfg.d_model;
            const float* v_ptr = qkv.data.data() + 2 * cfg.d_model;
            lc.append(k_ptr, v_ptr, cfg.d_model);
        }

        // Now run the full multi-token attention (uses gpt2.h's attention())
        Tensor attn_out = attention(ln1_out, bw, cfg);

        // Residual + LN2 + FFN + Residual
        Tensor x2({seq_len, cfg.d_model});
        for (int i = 0; i < x.num_elements(); ++i)
            x2.data[i] = x.data[i] + attn_out.data[i];

        Tensor ln2_out({seq_len, cfg.d_model});
        for (int t = 0; t < seq_len; ++t) {
            Tensor xt = row(x2, t);
            layernorm(xt, bw.ln2_gamma, bw.ln2_beta);
            set_row(ln2_out, t, xt);
        }

        Tensor ffn_out = ffn(ln2_out, bw, cfg);
        Tensor x3({seq_len, cfg.d_model});
        for (int i = 0; i < x2.num_elements(); ++i)
            x3.data[i] = x2.data[i] + ffn_out.data[i];

        x = std::move(x3);
    }

    // Final LayerNorm on the last token only (we only need the next-token prediction)
    Tensor last_hidden = row(x, seq_len - 1);
    layernorm(last_hidden, weights.ln_f_gamma, weights.ln_f_beta);

    // LM head: last_hidden @ wte.T -> logits {vocab_size}
    Tensor logits({cfg.vocab_size});
    for (int v = 0; v < cfg.vocab_size; ++v) {
        float dot = 0.0f;
        for (int d = 0; d < cfg.d_model; ++d)
            dot += last_hidden.data[d] * weights.wte.data[v * cfg.d_model + d];
        logits.data[v] = dot;
    }
    return logits;
}


// ===========================================================================
// decode_step() — generate ONE new token using the KV-cache.
// ===========================================================================
// token_id: the previously generated (or last prompt) token
// pos:      its absolute position in the sequence
// Returns: logits {vocab_size} for the NEXT token
inline Tensor decode_step(int token_id, int pos,
                            const GPT2Weights& weights,
                            const GPT2Config& cfg,
                            KVCache& cache) {
    assert(pos < cfg.max_seq);

    // Build embedding for this single token
    Tensor x({cfg.d_model});
    for (int d = 0; d < cfg.d_model; ++d)
        x.data[d] = weights.wte.data[token_id * cfg.d_model + d] +
                    weights.wpe.data[pos        * cfg.d_model + d];

    // Pass through all transformer blocks (single-token path, cached)
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        x = transformer_block_cached(x, weights.blocks[layer],
                                     cache.layers[layer], cfg);
    }

    // Final LayerNorm
    layernorm(x, weights.ln_f_gamma, weights.ln_f_beta);

    // LM head -> logits {vocab_size}
    Tensor logits({cfg.vocab_size});
    for (int v = 0; v < cfg.vocab_size; ++v) {
        float dot = 0.0f;
        for (int d = 0; d < cfg.d_model; ++d)
            dot += x.data[d] * weights.wte.data[v * cfg.d_model + d];
        logits.data[v] = dot;
    }
    return logits;
}


// ===========================================================================
// Decoding strategies
// ===========================================================================

// --- Greedy: always pick the most likely token ---
inline int greedy(const Tensor& logits) {
    return (int)(std::max_element(logits.data.begin(), logits.data.end())
                 - logits.data.begin());
}

// --- Top-k sampling: sample from the k most likely tokens ---
// temperature: divide logits by this before softmax (>1 = more random, <1 = sharper)
inline int top_k_sample(const Tensor& logits, int k, float temperature,
                         std::mt19937& rng) {
    assert(k >= 1);
    int V = logits.shape[0];
    k = std::min(k, V);

    // Build (logit, index) pairs and partial-sort to find top-k
    std::vector<std::pair<float,int>> pairs(V);
    for (int i = 0; i < V; ++i) pairs[i] = {logits.data[i], i};
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    // Apply temperature and softmax over top-k
    Tensor top_logits(std::vector<float>(k, 0.0f), {k});
    for (int i = 0; i < k; ++i)
        top_logits.data[i] = pairs[i].first / temperature;
    softmax(top_logits);

    // Sample
    std::discrete_distribution<int> dist(top_logits.data.begin(), top_logits.data.end());
    return pairs[dist(rng)].second;
}

// --- Top-p (nucleus) sampling: sample from the smallest set whose cumulative probability ≥ p ---
inline int top_p_sample(const Tensor& logits, float p, float temperature,
                         std::mt19937& rng) {
    int V = logits.shape[0];

    // Sort by logit descending
    std::vector<std::pair<float,int>> pairs(V);
    for (int i = 0; i < V; ++i) pairs[i] = {logits.data[i] / temperature, i};
    std::sort(pairs.begin(), pairs.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    // Compute softmax probabilities
    std::vector<float> probs(V);
    float max_l = pairs[0].first;
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) {
        probs[i] = std::exp(pairs[i].first - max_l);
        sum += probs[i];
    }
    for (float& f : probs) f /= sum;

    // Find nucleus: smallest prefix with cumulative probability ≥ p
    float cumul = 0.0f;
    int nucleus_size = 1;
    for (int i = 0; i < V; ++i) {
        cumul += probs[i];
        nucleus_size = i + 1;
        if (cumul >= p) break;
    }

    // Sample from the nucleus
    Tensor nucleus_probs(std::vector<float>(probs.begin(),
                                            probs.begin() + nucleus_size),
                         {nucleus_size});
    // Re-normalise
    float ns = 0.0f;
    for (float f : nucleus_probs.data) ns += f;
    for (float& f : nucleus_probs.data) f /= ns;

    std::discrete_distribution<int> dist(nucleus_probs.data.begin(),
                                          nucleus_probs.data.end());
    return pairs[dist(rng)].second;
}