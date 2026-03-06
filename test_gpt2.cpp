// test_gpt2.cpp — Milestone 2 test suite
// Run: g++ -std=c++17 -O2 -o test_gpt2 test_gpt2.cpp && ./test_gpt2
//
// We test with a TINY config (tiny vocab, 2 layers, 4 heads, d_model=64)
// so the tests run in milliseconds and are easy to reason about.
//
// What we verify:
//   1. Weight allocation shapes are correct
//   2. Forward pass produces logits of exactly the right shape
//   3. Causal mask is working (token 0 logits ≠ token 1 logits, etc.)
//   4. LM head produces a proper distribution after softmax
//   5. The full GPT-2 small config instantiates without crashing

#include "gpt2.h"
#include <cstdio>
#include <cmath>
#include <algorithm>  // std::max_element

// ---------------------------------------------------------------------------
// Tiny test framework (same as Milestone 1)
// ---------------------------------------------------------------------------
static int tests_run    = 0;
static int tests_passed = 0;

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

#define EXPECT_EQ(val, expected, label)                                        \
    do {                                                                        \
        ++tests_run;                                                            \
        if ((val) == (expected)) {                                              \
            ++tests_passed;                                                     \
            std::printf("  \033[32m[PASS]\033[0m %s = %d\n",                  \
                        (label), (int)(val));                                   \
        } else {                                                                \
            std::printf("  \033[31m[FAIL]\033[0m %s: got %d, expected %d\n",  \
                        (label), (int)(val), (int)(expected));                  \
        }                                                                       \
    } while (0)

#define EXPECT_TRUE(cond, label)                                               \
    do {                                                                        \
        ++tests_run;                                                            \
        if (cond) {                                                             \
            ++tests_passed;                                                     \
            std::printf("  \033[32m[PASS]\033[0m %s\n", (label));             \
        } else {                                                                \
            std::printf("  \033[31m[FAIL]\033[0m %s\n", (label));             \
        }                                                                       \
    } while (0)

#define EXPECT_NEAR(val, expected, label)                                      \
    do {                                                                        \
        ++tests_run;                                                            \
        if (approx_eq((val), (expected))) {                                    \
            ++tests_passed;                                                     \
            std::printf("  \033[32m[PASS]\033[0m %s = %.6f\n",                \
                        (label), (float)(val));                                 \
        } else {                                                                \
            std::printf("  \033[31m[FAIL]\033[0m %s: got %.6f expected %.6f\n",\
                        (label), (float)(val), (float)(expected));              \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Tiny config: small enough to run in <1s, large enough to be meaningful
// ---------------------------------------------------------------------------
// d_model=64, n_heads=4, d_head=16, d_ff=256, n_layers=2, vocab_size=100
GPT2Config tiny_cfg() {
    GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq    = 16;
    cfg.d_model    = 64;
    cfg.n_heads    = 4;
    cfg.n_layers   = 2;
    cfg.d_ff       = 256;   // 4 * 64
    cfg.d_head     = 16;    // 64 / 4
    return cfg;
}

// ---------------------------------------------------------------------------
// TEST 1 — Weight shapes
// ---------------------------------------------------------------------------
// Every tensor must have exactly the shape specified by the architecture.
// Shape bugs are silent: wrong shapes just silently produce garbage outputs.
void test_weight_shapes() {
    std::printf("\n--- test_weight_shapes ---\n");
    auto cfg = tiny_cfg();
    auto w = make_random_weights(cfg);

    // Token & position embeddings
    EXPECT_EQ(w.wte.shape[0], cfg.vocab_size, "wte rows (vocab_size)");
    EXPECT_EQ(w.wte.shape[1], cfg.d_model,    "wte cols (d_model)");
    EXPECT_EQ(w.wpe.shape[0], cfg.max_seq,    "wpe rows (max_seq)");
    EXPECT_EQ(w.wpe.shape[1], cfg.d_model,    "wpe cols (d_model)");

    // Block 0 weights (representative sample)
    const auto& b0 = w.blocks[0];
    EXPECT_EQ(b0.ln1_gamma.shape[0],  cfg.d_model,         "ln1_gamma size");
    EXPECT_EQ(b0.c_attn_w.shape[0],   3 * cfg.d_model,     "c_attn_w rows (out=3*d_model)");
    EXPECT_EQ(b0.c_attn_w.shape[1],   cfg.d_model,         "c_attn_w cols (in=d_model)");
    EXPECT_EQ(b0.c_attn_b.shape[0],   3 * cfg.d_model,     "c_attn_b size");
    EXPECT_EQ(b0.c_proj_w.shape[0],   cfg.d_model,         "c_proj_w rows");
    EXPECT_EQ(b0.c_proj_w.shape[1],   cfg.d_model,         "c_proj_w cols");
    EXPECT_EQ(b0.mlp_fc_w.shape[0],   cfg.d_ff,            "mlp_fc_w rows (out=d_ff)");
    EXPECT_EQ(b0.mlp_fc_w.shape[1],   cfg.d_model,         "mlp_fc_w cols (in=d_model)");
    EXPECT_EQ(b0.mlp_proj_w.shape[0], cfg.d_model,         "mlp_proj_w rows (out=d_model)");
    EXPECT_EQ(b0.mlp_proj_w.shape[1], cfg.d_ff,            "mlp_proj_w cols (in=d_ff)");

    // Number of blocks
    EXPECT_EQ((int)w.blocks.size(), cfg.n_layers, "number of transformer blocks");
}

// ---------------------------------------------------------------------------
// TEST 2 — Forward pass output shape
// ---------------------------------------------------------------------------
// gpt2_forward() must return logits of shape {seq_len, vocab_size}.
// Any shape error here would crash or produce garbage in Milestone 3.
void test_forward_shape() {
    std::printf("\n--- test_forward_shape ---\n");
    auto cfg = tiny_cfg();
    auto w = make_random_weights(cfg);

    // Feed 5 tokens
    std::vector<int> tokens = {0, 1, 2, 3, 4};
    Tensor logits = gpt2_forward(tokens, w, cfg);

    EXPECT_EQ(logits.ndim(),    2,                "logits ndim");
    EXPECT_EQ(logits.shape[0],  (int)tokens.size(), "logits rows (seq_len)");
    EXPECT_EQ(logits.shape[1],  cfg.vocab_size,   "logits cols (vocab_size)");
}

// ---------------------------------------------------------------------------
// TEST 3 — Causal masking
// ---------------------------------------------------------------------------
// Token at position 0 sees only itself.
// Token at position 1 sees positions 0 and 1.
// Therefore: the logits for position 0 and position 1 should differ.
// If the mask were broken they would be identical.
//
// We also verify that different input sequences produce different logits
// (i.e., the model is actually doing something, not returning constants).
void test_causal_masking() {
    std::printf("\n--- test_causal_masking ---\n");
    auto cfg = tiny_cfg();
    auto w = make_random_weights(cfg);

    std::vector<int> tokens = {3, 7, 2};
    Tensor logits = gpt2_forward(tokens, w, cfg);

    // Logit for token 0 and token 1 at vocab entry 0 should differ
    float l0 = logits.data[0 * cfg.vocab_size + 0];  // position 0, vocab 0
    float l1 = logits.data[1 * cfg.vocab_size + 0];  // position 1, vocab 0
    float l2 = logits.data[2 * cfg.vocab_size + 0];  // position 2, vocab 0

    EXPECT_TRUE(std::fabs(l0 - l1) > 1e-6f,
                "pos 0 and pos 1 logits differ (causal mask active)");
    EXPECT_TRUE(std::fabs(l1 - l2) > 1e-6f,
                "pos 1 and pos 2 logits differ (causal mask active)");

    // Verify: change only the last token → only last position's logits change
    std::vector<int> tokens_alt = {3, 7, 5};  // last token changed: 2 → 5
    Tensor logits_alt = gpt2_forward(tokens_alt, w, cfg);

    // Position 0 logits must be identical (it only sees token 0, which is same)
    float l0_alt = logits_alt.data[0 * cfg.vocab_size + 0];
    EXPECT_NEAR(l0, l0_alt, "pos 0 logits unchanged when only pos 2 token changes");

    // Position 2 logits must differ (different token at pos 2)
    float l2_alt = logits_alt.data[2 * cfg.vocab_size + 0];
    EXPECT_TRUE(std::fabs(l2 - l2_alt) > 1e-6f,
                "pos 2 logits change when pos 2 token changes");
}

// ---------------------------------------------------------------------------
// TEST 4 — Logits produce valid probability distribution
// ---------------------------------------------------------------------------
// After softmax, the logit row for any token position should:
//   - sum to 1.0
//   - all values in (0, 1)
//   - argmax = "predicted next token" (greedy decoding)
void test_softmax_over_logits() {
    std::printf("\n--- test_softmax_over_logits ---\n");
    auto cfg = tiny_cfg();
    auto w = make_random_weights(cfg);

    std::vector<int> tokens = {0, 5};
    Tensor logits = gpt2_forward(tokens, w, cfg);

    // Softmax the last token's logit row
    int last = (int)tokens.size() - 1;
    Tensor last_logits(
        std::vector<float>(logits.data.begin() + last * cfg.vocab_size,
                           logits.data.begin() + last * cfg.vocab_size + cfg.vocab_size),
        {cfg.vocab_size}
    );
    softmax(last_logits);

    // Sum should be 1.0
    float sum = 0.0f;
    for (float v : last_logits.data) sum += v;
    EXPECT_NEAR(sum, 1.0f, "softmax(logits) sums to 1.0");

    // All values should be positive
    bool all_positive = true;
    for (float v : last_logits.data)
        if (v <= 0.0f) { all_positive = false; break; }
    EXPECT_TRUE(all_positive, "all softmax probabilities > 0");

    // Greedy decode: find argmax
    int argmax = (int)(std::max_element(last_logits.data.begin(),
                                        last_logits.data.end())
                       - last_logits.data.begin());
    std::printf("  Greedy predicted next token id: %d  (prob=%.4f)\n",
                argmax, last_logits.data[argmax]);
    EXPECT_TRUE(argmax >= 0 && argmax < cfg.vocab_size,
                "greedy argmax is a valid token id");
}

// ---------------------------------------------------------------------------
// TEST 5 — Full GPT-2 small config instantiates
// ---------------------------------------------------------------------------
// We don't run a forward pass (too slow with d_model=768 and no BLAS),
// but we verify weight allocation succeeds and shapes are right.
void test_gpt2_small_shapes() {
    std::printf("\n--- test_gpt2_small_config ---\n");

    GPT2Config cfg;  // default = GPT-2 small
    cfg.validate();  // will assert if misconfigured

    auto w = make_random_weights(cfg);

    EXPECT_EQ(w.wte.shape[0], 50257, "GPT-2 small: vocab_size=50257");
    EXPECT_EQ(w.wte.shape[1], 768,   "GPT-2 small: d_model=768");
    EXPECT_EQ((int)w.blocks.size(), 12, "GPT-2 small: 12 transformer blocks");
    EXPECT_EQ(w.blocks[0].c_attn_w.shape[0], 3 * 768, "c_attn_w rows = 3*768 (out_dim)");
    EXPECT_EQ(w.blocks[0].mlp_fc_w.shape[0], 3072,    "mlp_fc_w rows = 3072 (d_ff)");

    // Total parameters (rough check)
    size_t total_params = 0;
    total_params += w.wte.num_elements();
    total_params += w.wpe.num_elements();
    for (const auto& b : w.blocks) {
        total_params += b.ln1_gamma.num_elements() + b.ln1_beta.num_elements();
        total_params += b.c_attn_w.num_elements()  + b.c_attn_b.num_elements();
        total_params += b.c_proj_w.num_elements()  + b.c_proj_b.num_elements();
        total_params += b.ln2_gamma.num_elements() + b.ln2_beta.num_elements();
        total_params += b.mlp_fc_w.num_elements()  + b.mlp_fc_b.num_elements();
        total_params += b.mlp_proj_w.num_elements()+ b.mlp_proj_b.num_elements();
    }
    total_params += w.ln_f_gamma.num_elements() + w.ln_f_beta.num_elements();
    // Note: LM head is weight-tied (wte reused), so not counted again.

    std::printf("  Total allocated parameters: %zu\n", total_params);
    // GPT-2 small has ~124M params. Our count excludes the LM head (weight-tied)
    // so it should be slightly less. ~85M is the transformer body alone.
    EXPECT_TRUE(total_params > 80'000'000UL,
                "param count in expected range (>80M)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== Milestone 2: GPT-2 Forward Pass ===\n");
    std::printf("GPT-2 Inference Engine — from scratch in C++\n");

    test_weight_shapes();
    test_forward_shape();
    test_causal_masking();
    test_softmax_over_logits();
    test_gpt2_small_shapes();

    std::printf("\n=== Results: %d / %d tests passed ===\n",
                tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}