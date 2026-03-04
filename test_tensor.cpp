// test_tensor.cpp — test suite for checkpt 1: Tensor + Math primitives 

#include "tensor.h"
#include <cstdio>
#include <cmath>

static int tests_run    = 0;
static int tests_passed = 0;

// Check that two floats are close enough (relative to a small epsilon).
static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// Assert that val is approximately equal to expected.
#define EXPECT_NEAR(val, expected, label)                                 \
    do {                                                                   \
        ++tests_run;                                                       \
        if (approx_eq((val), (expected))) {                               \
            ++tests_passed;                                                \
            std::printf("  \033[32m[PASS]\033[0m %s = %.6f\n",           \
                        (label), (float)(val));                            \
        } else {                                                           \
            std::printf("  \033[31m[FAIL]\033[0m %s: got %.6f, "         \
                        "expected %.6f\n",                                 \
                        (label), (float)(val), (float)(expected));        \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// TEST 1 — matmul
// ---------------------------------------------------------------------------
// lets say we multiply a 2×3 matrix by a 3×2 matrix:
//
//   A = | 1 2 3 |     B = | 7  8  |
//       | 4 5 6 |         | 9  10 |
//                         | 11 12 |
//
//   C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
//   C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
//   C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
//   C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
// ---------------------------------------------------------------------------
void test_matmul() {
    std::printf("\n--- test_matmul ---\n");

    Tensor A({std::vector<float>{1,2,3, 4,5,6}, {2, 3}});
    Tensor B({std::vector<float>{7,8, 9,10, 11,12}, {3, 2}});

    Tensor C = matmul(A, B);

    EXPECT_NEAR(C.at(0, 0), 58.0f,  "C[0][0]");
    EXPECT_NEAR(C.at(0, 1), 64.0f,  "C[0][1]");
    EXPECT_NEAR(C.at(1, 0), 139.0f, "C[1][0]");
    EXPECT_NEAR(C.at(1, 1), 154.0f, "C[1][1]");
}

// ---------------------------------------------------------------------------
// TEST 2 — softmax
// ---------------------------------------------------------------------------
// Input: [1.0, 2.0, 3.0]
//
// Step 1: max = 3.0
// Step 2: exp([1-3, 2-3, 3-3]) = exp([-2, -1, 0]) ≈ [0.1353, 0.3679, 1.0]
// Step 3: sum ≈ 1.5032
// Result ≈ [0.0900, 0.2447, 0.6652]  (must sum to 1.0)
//
// Key properties to verify:
//   - All values are in (0, 1)
//   - Values sum to exactly 1.0
// ---------------------------------------------------------------------------
void test_softmax() {
    std::printf("\n--- test_softmax ---\n");

    Tensor x({std::vector<float>{1.0f, 2.0f, 3.0f}, {3}});
    softmax(x);

    // Known expected values (computed with numpy: scipy.special.softmax)
    EXPECT_NEAR(x.at(0), 0.09003057f, "softmax[0]");
    EXPECT_NEAR(x.at(1), 0.24472847f, "softmax[1]");
    EXPECT_NEAR(x.at(2), 0.66524096f, "softmax[2]");

    // Sum must be 1.0
    float s = x.data[0] + x.data[1] + x.data[2];
    EXPECT_NEAR(s, 1.0f, "softmax sum");

    // Stability test: large inputs should not produce NaN/inf
    Tensor large({std::vector<float>{1000.0f, 1001.0f, 1002.0f}, {3}});
    softmax(large);
    float s2 = large.data[0] + large.data[1] + large.data[2];
    EXPECT_NEAR(s2, 1.0f, "softmax sum (large inputs)");
}

// ---------------------------------------------------------------------------
// TEST 3 — layernorm
// ---------------------------------------------------------------------------
// Input: [1.0, 2.0, 3.0, 4.0]
// gamma = [1, 1, 1, 1]  (no scaling)
// beta  = [0, 0, 0, 0]  (no shift)
//
// With identity gamma/beta, layernorm should produce zero-mean, unit-var output.
//
// mean = (1+2+3+4)/4 = 2.5
// var  = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
//       = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
// std  = sqrt(1.25 + 1e-5) ≈ 1.11803
// y    = [(1-2.5)/1.118, (2-2.5)/1.118, (3-2.5)/1.118, (4-2.5)/1.118]
//       ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
// ---------------------------------------------------------------------------
void test_layernorm() {
    std::printf("\n--- test_layernorm ---\n");

    Tensor x({std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}});
    Tensor gamma({std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, {4}});
    Tensor beta ({std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, {4}});

    layernorm(x, gamma, beta);

    EXPECT_NEAR(x.at(0), -1.3416407f, "layernorm[0]");
    EXPECT_NEAR(x.at(1), -0.4472136f, "layernorm[1]");
    EXPECT_NEAR(x.at(2),  0.4472136f, "layernorm[2]");
    EXPECT_NEAR(x.at(3),  1.3416407f, "layernorm[3]");

    // After normalisation the mean should be ≈ 0 and std ≈ 1
    float mean = (x.data[0] + x.data[1] + x.data[2] + x.data[3]) / 4.0f;
    EXPECT_NEAR(mean, 0.0f, "layernorm output mean");

    float var = 0.0f;
    for (float v : x.data) var += (v - mean) * (v - mean);
    var /= 4.0f;
    EXPECT_NEAR(var, 1.0f, "layernorm output variance (approx)");

    // Test with non-identity gamma/beta: scale by 2, shift by 1
    Tensor x2({std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}});
    Tensor gamma2({std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f}, {4}});
    Tensor beta2 ({std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, {4}});
    layernorm(x2, gamma2, beta2);
    // gamma=2, beta=1 should give: 2 * normalised + 1
    EXPECT_NEAR(x2.at(0), 2.0f * -1.3416407f + 1.0f, "layernorm[0] scaled");
    EXPECT_NEAR(x2.at(3), 2.0f *  1.3416407f + 1.0f, "layernorm[3] scaled");
}

// ---------------------------------------------------------------------------
// TEST 4 — gelu
// ---------------------------------------------------------------------------
// GELU at a few known points:
//   GELU(0.0) = 0.0          (by symmetry of the Gaussian)
//   GELU(1.0) ≈ 0.8413       (≈ Φ(1) * 1)
//   GELU(-1.0) ≈ -0.1587     (negative values are suppressed)
//   GELU(2.0) ≈ 1.9545
//
// Expected values cross-checked with PyTorch:
//   import torch; torch.nn.functional.gelu(torch.tensor([0., 1., -1., 2.]))
// ---------------------------------------------------------------------------
void test_gelu() {
    std::printf("\n--- test_gelu ---\n");

    Tensor x({std::vector<float>{0.0f, 1.0f, -1.0f, 2.0f}, {4}});
    gelu(x);

    // GPT-2 uses the tanh *approximation* of GELU, not the exact erf-based form.
    // The approximation differs from PyTorch's default gelu by ~1e-4, so we use
    // a slightly looser tolerance (5e-4) here to reflect the approximation error.
    EXPECT_NEAR(x.at(0),  0.0f,       "gelu(0.0)");
    // 0.841192 is the correct tanh-approx GELU(1.0); PyTorch exact gives 0.8413
    EXPECT_NEAR(x.at(1),  0.841192f,  "gelu(1.0)");
    EXPECT_NEAR(x.at(2), -0.158808f,  "gelu(-1.0)");
    EXPECT_NEAR(x.at(3),  1.9545f,    "gelu(2.0)");

    // GELU should always be ≤ x for positive x (it's a "soft" gate, not identity)
    // and should be close to x for large positive x
    Tensor large({std::vector<float>{10.0f}, {1}});
    gelu(large);
    EXPECT_NEAR(large.at(0), 10.0f, "gelu(10.0) ≈ 10");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("GPT-2 Inference Engine from scratch in C++\n");
    std::printf("---------- Milestone 1: Tensor & Math Primitives ----------\n");

    test_matmul();
    test_softmax();
    test_layernorm();
    test_gelu();

    std::printf("\n=== Results: %d / %d tests passed ===\n",
                tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}