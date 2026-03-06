#pragma once
// gpt2.h: GPT-2 (small) forward pass
//
// Architecture source: "Language Models are Unsupervised Multitask Learners"
//                       Radford et al., OpenAI 2019
// Hyperparameter reference: https://keras.io/keras_hub/api/models/gpt2/gpt2_backbone/
//
// This file contains:
//   1. GPT2Config  — all hyperparameters in one place
//   2. GPT2Weights — every weight tensor the model needs
//   3. attention() — masked multi-head self-attention
//   4. ffn()       — position-wise feed-forward network
//   5. transformer_block() — one full decoder block
//   6. gpt2_forward()      — the complete forward pass
//
// NO weight loading in M2. Weights initialised to random/small values

#include "tensor.h"
#include <random>
#include <cstring>  // memcpy
#include <limits>   // numeric_limits

// ===========================================================================
// 1. GPT2Config — all architecture constants in one struct
// ===========================================================================
//
// GPT-2 Small:
//   vocab_size = 50257   — BPE vocabulary size
//   max_seq    = 1024    — maximum number of tokens the model can process
//   d_model    = 768     — "hidden size": every token is a 768-dim vector
//   n_heads    = 12      — number of parallel attention heads
//   n_layers   = 12      — number of stacked transformer blocks
//   d_ff       = 3072    — FFN inner dimension = 4 * d_model
//   d_head     = 64      — per-head dimension = d_model / n_heads
struct GPT2Config {
    int vocab_size = 50257;
    int max_seq    = 1024;
    int d_model    = 768;
    int n_heads    = 12;
    int n_layers   = 12;
    int d_ff       = 3072;  // 4 * d_model
    int d_head     = 64;    // d_model / n_heads

    // Sanity-check the config: make misconfiguration fail loudly at startup
    void validate() const {
        assert(d_model == n_heads * d_head &&
               "d_model must equal n_heads * d_head");
        assert(d_ff == 4 * d_model &&
               "d_ff must equal 4 * d_model");
    }
};


// ===========================================================================
// 2. GPT2Weights — every learnable parameter the model needs
// ===========================================================================
// In a proper inference engine these would be loaded from disk (will do in Milestone 3).
// Here they are zero/random-initialised just to verify shapes are correct.
//
// Naming follows OpenAI's weight file conventions so Milestone 3 can map
// names 1-to-1:
//   wte  = word token embeddings     {vocab_size, d_model}
//   wpe  = word position embeddings  {max_seq, d_model}
//   Each transformer block has:
//     ln_1     — LayerNorm before attention
//     c_attn   — combined Q,K,V projection (3*d_model output)
//     c_proj   — attention output projection
//     ln_2     — LayerNorm before FFN
//     mlp_fc   — FFN first linear (d_model -> d_ff)
//     mlp_proj — FFN second linear (d_ff -> d_model)
//   ln_f = final LayerNorm after all blocks
//
// NOTE: GPT-2 ties the LM head weights to wte (weight tying):
//   logits = x @ wte.T    (no separate lm_head weight)

struct BlockWeights {
    // LayerNorm 1 (before attention): scale (gamma) and shift (beta)
    // Both shape: {d_model}
    Tensor ln1_gamma, ln1_beta;

    // Combined QKV projection: W shape {3*d_model, d_model}, b shape {3*d_model}
    Tensor c_attn_w, c_attn_b;

    // Attention output projection: W {d_model, d_model}, b {d_model}
    Tensor c_proj_w, c_proj_b;

    // LayerNorm 2 (before FFN)
    Tensor ln2_gamma, ln2_beta;

    // FFN: first linear  d_model -> d_ff,  W {d_ff, d_model}, b {d_ff}
    Tensor mlp_fc_w, mlp_fc_b;

    // FFN: second linear  d_ff -> d_model, W {d_model, d_ff}, b {d_model}
    Tensor mlp_proj_w, mlp_proj_b;
};

struct GPT2Weights {
    Tensor wte;  // token embeddings  {vocab_size, d_model}
    Tensor wpe;  // position embeddings {max_seq, d_model}

    std::vector<BlockWeights> blocks;  // one per layer

    Tensor ln_f_gamma, ln_f_beta;  // final LayerNorm {d_model}
};


// ===========================================================================
// Helper: random-initialise a Tensor (small Gaussian noise)
// ===========================================================================
// We use a fixed seed so test output is reproducible.
static void rand_init(Tensor& t, float std_dev = 0.02f, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (float& v : t.data)
        v = dist(rng);
}

// ===========================================================================
// GPT2Weights factory: allocate and randomly-initialise all tensors
// ===========================================================================
// This is the M2 substitute for real weight loading (Milestone 3).
// It lets us call gpt2_forward() and check that shapes flow correctly.
inline GPT2Weights make_random_weights(const GPT2Config& cfg) {
    cfg.validate();
    GPT2Weights w;

    // Embeddings
    w.wte = Tensor({cfg.vocab_size, cfg.d_model});
    rand_init(w.wte, 0.02f, 1);

    w.wpe = Tensor({cfg.max_seq, cfg.d_model});
    rand_init(w.wpe, 0.01f, 2);

    // Transformer blocks
    w.blocks.resize(cfg.n_layers);
    unsigned seed = 10;
    for (int i = 0; i < cfg.n_layers; ++i) {
        BlockWeights& b = w.blocks[i];
        // LayerNorm 1 - gamma initialised to 1, beta to 0 (identity)
        b.ln1_gamma = Tensor(std::vector<float>(cfg.d_model, 1.0f), {cfg.d_model});
        b.ln1_beta  = Tensor(std::vector<float>(cfg.d_model, 0.0f), {cfg.d_model});

        // QKV projection: maps {d_model} -> {3*d_model}, so W is {3*d_model, d_model}
        // matvec(W, x) computes W @ x, requiring W.shape = {out, in}
        b.c_attn_w = Tensor({3 * cfg.d_model, cfg.d_model});
        rand_init(b.c_attn_w, 0.02f, seed++);
        b.c_attn_b = Tensor({3 * cfg.d_model});

        // Attention output projection: {d_model} -> {d_model}, W is {d_model, d_model}
        b.c_proj_w = Tensor({cfg.d_model, cfg.d_model});
        rand_init(b.c_proj_w, 0.02f, seed++);
        b.c_proj_b = Tensor({cfg.d_model});

        // LayerNorm 2
        b.ln2_gamma = Tensor(std::vector<float>(cfg.d_model, 1.0f), {cfg.d_model});
        b.ln2_beta  = Tensor(std::vector<float>(cfg.d_model, 0.0f), {cfg.d_model});

        // FFN first linear: {d_model} -> {d_ff}, W is {d_ff, d_model}
        b.mlp_fc_w = Tensor({cfg.d_ff, cfg.d_model});
        rand_init(b.mlp_fc_w, 0.02f, seed++);
        b.mlp_fc_b = Tensor({cfg.d_ff});

        // FFN second linear: {d_ff} -> {d_model}, W is {d_model, d_ff}
        b.mlp_proj_w = Tensor({cfg.d_model, cfg.d_ff});
        rand_init(b.mlp_proj_w, 0.02f, seed++);
        b.mlp_proj_b = Tensor({cfg.d_model});
    }

    // Final LayerNorm
    w.ln_f_gamma = Tensor(std::vector<float>(cfg.d_model, 1.0f), {cfg.d_model});
    w.ln_f_beta  = Tensor(std::vector<float>(cfg.d_model, 0.0f), {cfg.d_model});

    return w;
}


// ===========================================================================
// Helper: extract row t from a 2D tensor (returns a copy as a 1D tensor)
// ===========================================================================
// e.g. row(wte, token_id)  ->  the 768-dim embedding for that token
inline Tensor row(const Tensor& mat, int t) {
    assert(mat.ndim() == 2);
    int cols = mat.shape[1];
    // The row starts at offset t * cols in the flat data array.
    return Tensor(
        std::vector<float>(mat.data.begin() + t * cols,
                           mat.data.begin() + t * cols + cols),
        {cols}
    );
}

// ===========================================================================
// Helper: set row t of a 2D tensor from a 1D tensor (in-place)
// ===========================================================================
inline void set_row(Tensor& mat, int t, const Tensor& vec) {
    assert(mat.ndim() == 2 && vec.ndim() == 1);
    int cols = mat.shape[1];
    assert(vec.shape[0] == cols);
    std::copy(vec.data.begin(), vec.data.end(),
              mat.data.begin() + t * cols);
}


// ===========================================================================
// 3. attention() — Masked Multi-Head Self-Attention
// ===========================================================================
// INPUT:  x  {seq_len, d_model}  — token representations for this layer
//         bw — weights for this block
//         cfg
// OUTPUT: {seq_len, d_model}   — attention-transformed token representations
//
// NOTE for self: A step-by-step walkthrough:
//
//   A) Project x -> Q, K, V  (combined matmul then slice)
//   B) Split Q, K, V across n_heads
//   C) For each head h:
//        scores = Q_h @ K_h.T / sqrt(d_head)    shape: {seq_len, seq_len}
//        apply causal mask (set future positions to -inf)
//        weights = softmax(scores)               shape: {seq_len, seq_len}
//        head_out = weights @ V_h                shape: {seq_len, d_head}
//   D) Concatenate all head outputs             shape: {seq_len, d_model}
//   E) Project back with c_proj                 shape: {seq_len, d_model}
// ===========================================================================
inline Tensor attention(const Tensor& x,
                        const BlockWeights& bw,
                        const GPT2Config& cfg) {
    int seq_len = x.shape[0];
    int d_model = cfg.d_model;
    int n_heads = cfg.n_heads;
    int d_head  = cfg.d_head;

    // -----------------------------------------------------------------------
    // A) Combined QKV projection
    // -----------------------------------------------------------------------
    // c_attn_w has shape {d_model, 3*d_model}.
    // For each token position t, we compute:
    //   qkv[t] = x[t] @ c_attn_w + c_attn_b    ->  a vector of length 3*d_model
    //
    // The first d_model values are Q, next d_model are K, last d_model are V.
    // We store all seq_len token projections in a {seq_len, 3*d_model} tensor.

    Tensor qkv({seq_len, 3 * d_model});  // zero-initialised

    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(x, t);                       // shape: {d_model}
        Tensor proj = matvec(bw.c_attn_w, xt);       // shape: {3*d_model}
        add_bias(proj, bw.c_attn_b);
        set_row(qkv, t, proj);
    }

    // -----------------------------------------------------------------------
    // B+C) Multi-head attention
    // -----------------------------------------------------------------------
    // Process each head independently, then concatenate.
    // Each head works on a slice of the QKV vectors:
    //   head h uses columns [h*d_head, (h+1)*d_head) of Q, K, V respectively.
    //
    // Output: {seq_len, d_model} (= n_heads × d_head = d_model)

    Tensor attn_out({seq_len, d_model});  // will be filled head-by-head

    for (int h = 0; h < n_heads; ++h) {
        // Offsets into the flat QKV: Q starts at 0, K at d_model, V at 2*d_model
        int q_off = h * d_head;
        int k_off = d_model + h * d_head;
        int v_off = 2 * d_model + h * d_head;

        // Build Q_h, K_h, V_h  shapes: {seq_len, d_head}
        // We copy the relevant slice from each row of qkv.
        Tensor Q_h({seq_len, d_head});
        Tensor K_h({seq_len, d_head});
        Tensor V_h({seq_len, d_head});

        for (int t = 0; t < seq_len; ++t) {
            // qkv row t starts at offset t * (3*d_model) in qkv.data
            const float* qkv_t = qkv.data.data() + t * 3 * d_model;
            float* q_row = Q_h.data.data() + t * d_head;
            float* k_row = K_h.data.data() + t * d_head;
            float* v_row = V_h.data.data() + t * d_head;
            std::memcpy(q_row, qkv_t + q_off, d_head * sizeof(float));
            std::memcpy(k_row, qkv_t + k_off, d_head * sizeof(float));
            std::memcpy(v_row, qkv_t + v_off, d_head * sizeof(float));
        }

        // --- Attention scores: scores = Q_h @ K_h.T / sqrt(d_head) ---
        // scores[i][j] = how much token i attends to token j
        // Shape: {seq_len, seq_len}

        float scale = 1.0f / std::sqrt((float)d_head);

        Tensor scores({seq_len, seq_len});
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d_head; ++k)
                    dot += Q_h.data[i * d_head + k] * K_h.data[j * d_head + k];
                scores.data[i * seq_len + j] = dot * scale;
            }
        }

        // --- Autoregressive or Causal mask ---
        // GPT-2 is a decoder-only model: token at position i should only
        // attend to positions ≤ i (it should not "see" future tokens)
        //
        // Enforce this by setting scores[i][j] = -INF for all j > i.
        // After softmax, exp(-INF) = 0, so those positions contribute nothing
        // to the weighted sum.
    

        const float NEG_INF = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < seq_len; ++i)
            for (int j = i + 1; j < seq_len; ++j)
                scores.data[i * seq_len + j] = NEG_INF;

        // --- Softmax each row of scores independently ---
        // Each row i is a distribution over which past tokens token i attends to.
        for (int i = 0; i < seq_len; ++i) {
            // Wrap the row as a temporary 1D tensor to reuse our softmax().
            Tensor row_i(std::vector<float>(
                             scores.data.begin() + i * seq_len,
                             scores.data.begin() + i * seq_len + seq_len),
                         {seq_len});
            softmax(row_i);
            std::copy(row_i.data.begin(), row_i.data.end(),
                      scores.data.begin() + i * seq_len);
        }

        // --- Weighted sum of values: head_out = scores @ V_h ---
        // head_out[i] = sum_j(scores[i][j] * V_h[j])
        // Shape: {seq_len, d_head}
        Tensor head_out = matmul(scores, V_h);

        // --- Write head output into the correct columns of attn_out ---
        // Head h owns columns [h*d_head, (h+1)*d_head) of attn_out.
        for (int t = 0; t < seq_len; ++t) {
            float* dst = attn_out.data.data() + t * d_model + h * d_head;
            const float* src = head_out.data.data() + t * d_head;
            std::memcpy(dst, src, d_head * sizeof(float));
        }
    }

    // -----------------------------------------------------------------------
    // E) Output projection: attn_out @ c_proj_w + c_proj_b
    // -----------------------------------------------------------------------
    // This mixes information across heads. Shape stays {seq_len, d_model}.
    Tensor out({seq_len, d_model});
    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(attn_out, t);       // {d_model}
        Tensor proj = matvec(bw.c_proj_w, xt);
        add_bias(proj, bw.c_proj_b);
        set_row(out, t, proj);
    }
    return out;
}


// ===========================================================================
// 4. ffn() — Position-wise Feed-Forward Network
// ===========================================================================
// INPUT:  x  {seq_len, d_model}
// OUTPUT: {seq_len, d_model}
//
// Each token is processed independently through two linear layers:
//   hidden = GELU( x @ W_fc + b_fc )     {d_model} -> {d_ff=3072}
//   out    = hidden @ W_proj + b_proj     {d_ff}    -> {d_model}
// ===========================================================================
inline Tensor ffn(const Tensor& x,
                  const BlockWeights& bw,
                  const GPT2Config& cfg) {
    int seq_len = x.shape[0];
    Tensor out({seq_len, cfg.d_model});

    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(x, t);                    // {d_model}

        // First linear: expand to d_ff
        Tensor h = matvec(bw.mlp_fc_w, xt);      // {d_ff}
        add_bias(h, bw.mlp_fc_b);

        // GELU non-linearity (in-place)
        gelu(h);

        // Second linear: project back to d_model
        Tensor y = matvec(bw.mlp_proj_w, h);     // {d_model}
        add_bias(y, bw.mlp_proj_b);

        set_row(out, t, y);
    }
    return out;
}

// ===========================================================================
// 5. transformer_block() — One complete decoder block
// ===========================================================================
// INPUT:  x  {seq_len, d_model}
// OUTPUT: {seq_len, d_model}
//
// GPT-2 uses Pre-LN (LayerNorm BEFORE each sub-layer)
// The residual connections let gradients flow directly to earlier layers
//
// Pseudocode for self:
//   x = x + Attention( LayerNorm(x) )
//   x = x + FFN( LayerNorm(x) )
// ===========================================================================
inline Tensor transformer_block(const Tensor& x,
                                 const BlockWeights& bw,
                                 const GPT2Config& cfg) {
    int seq_len = x.shape[0];

    // --- Sub-layer 1: LayerNorm + Attention + Residual ---

    // 1a. LayerNorm each token's vector independently
    Tensor ln1_out({seq_len, cfg.d_model});
    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(x, t);
        layernorm(xt, bw.ln1_gamma, bw.ln1_beta);
        set_row(ln1_out, t, xt);
    }

    // 1b. Multi-head attention
    Tensor attn_out = attention(ln1_out, bw, cfg);

    // 1c. Residual: add input x back (element-wise)
    Tensor x2({seq_len, cfg.d_model});
    for (int i = 0; i < x.num_elements(); ++i)
        x2.data[i] = x.data[i] + attn_out.data[i];

    // --- Sub-layer 2: LayerNorm + FFN + Residual ---

    // 2a. LayerNorm
    Tensor ln2_out({seq_len, cfg.d_model});
    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(x2, t);
        layernorm(xt, bw.ln2_gamma, bw.ln2_beta);
        set_row(ln2_out, t, xt);
    }

    // 2b. FFN
    Tensor ffn_out = ffn(ln2_out, bw, cfg);

    // 2c. Residual
    Tensor x3({seq_len, cfg.d_model});
    for (int i = 0; i < x2.num_elements(); ++i)
        x3.data[i] = x2.data[i] + ffn_out.data[i];

    return x3;
}


// ===========================================================================
// 6. gpt2_forward() — The complete GPT-2 forward pass
// ===========================================================================
// INPUT:  token_ids — vector of integer token IDs, length = seq_len
//         weights   — all model parameters
//         cfg       — hyperparameters
// OUTPUT: logits    — Tensor of shape {seq_len, vocab_size}
//                     logits[t] is the unnormalised distribution over the
//                     next token after position t.
//
// Note for self: Full pipeline:
//   1. Look up token embeddings:    wte[token_ids]  -> {seq_len, d_model}
//   2. Add position embeddings:     wpe[0..seq_len] -> {seq_len, d_model}
//   3. Pass through n_layers transformer blocks
//   4. Final LayerNorm
//   5. LM head: x @ wte.T          -> {seq_len, vocab_size}
//      (weight-tied: same matrix as the token embedding table, transposed)
// ===========================================================================
inline Tensor gpt2_forward(const std::vector<int>& token_ids,
                            const GPT2Weights& weights,
                            const GPT2Config& cfg) {
    int seq_len = (int)token_ids.size();
    assert(seq_len > 0 && seq_len <= cfg.max_seq);

    // -----------------------------------------------------------------------
    // Step 1+2: Embeddings
    // -----------------------------------------------------------------------
    // For each position t, the input is:
    //   x[t] = wte[token_ids[t]]  +  wpe[t]
    //
    // wte: learned meaning of each token type ("what" the word means)
    // wpe: learned meaning of each position  ("where" in the sequence)

    Tensor x({seq_len, cfg.d_model});
    for (int t = 0; t < seq_len; ++t) {
        int tok = token_ids[t];
        assert(tok >= 0 && tok < cfg.vocab_size);
        for (int d = 0; d < cfg.d_model; ++d) {
            x.data[t * cfg.d_model + d] =
                weights.wte.data[tok * cfg.d_model + d] +   // token embedding
                weights.wpe.data[t   * cfg.d_model + d];    // position embedding
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Pass through all transformer blocks
    // -----------------------------------------------------------------------
    // Each block refines the representation. After 12 blocks the vectors
    // encode not just the token identity but rich contextual meaning.
    for (int i = 0; i < cfg.n_layers; ++i) {
        x = transformer_block(x, weights.blocks[i], cfg);
    }

    // -----------------------------------------------------------------------
    // Step 4: Final LayerNorm
    // -----------------------------------------------------------------------
    for (int t = 0; t < seq_len; ++t) {
        Tensor xt = row(x, t);
        layernorm(xt, weights.ln_f_gamma, weights.ln_f_beta);
        set_row(x, t, xt);
    }

    // -----------------------------------------------------------------------
    // Step 5: LM Head — project to vocabulary logits
    // -----------------------------------------------------------------------
    // logits = x @ wte.T
    //
    // wte has shape {vocab_size, d_model}.
    // wte.T has shape {d_model, vocab_size}.
    // x @ wte.T has shape {seq_len, vocab_size}.

    Tensor logits({seq_len, cfg.vocab_size});

    for (int t = 0; t < seq_len; ++t) {
        // x[t] has shape {d_model}; we dot it with every row of wte.
        for (int v = 0; v < cfg.vocab_size; ++v) {
            float dot = 0.0f;
            for (int d = 0; d < cfg.d_model; ++d)
                dot += x.data[t * cfg.d_model + d] *
                       weights.wte.data[v * cfg.d_model + d];
            logits.data[t * cfg.vocab_size + v] = dot;
        }
    }

    return logits;
}