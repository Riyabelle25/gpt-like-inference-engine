#pragma once
// tensor.h — A minimal Tensor class for GPT-2 inference.
// Stores data as a flat 1D vector of floats, with a shape that describes how many dimensions exist and how large each one is.
//
// Mental model:
//   A 2D tensor of shape {3, 4} is 12 floats in memory, laid out row-major:
//   [ row0_col0, row0_col1, ..., row2_col3 ]
//   To get element at (row=i, col=j): data[i * 4 + j]
//
// Why flat storage?
//   It matches how weight files are stored on disk (raw float32 blobs),
//   and makes pointer arithmetic for matmul straightforward.

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <numeric>    // std::accumulate
#include <functional> // std::multiplies
#include <cassert>

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
struct Tensor {
    std::vector<float> data;   // flat storage, row-major
    std::vector<int>   shape;  // e.g. {seq_len, d_model} for a 2D tensor

    // --- Constructors -------------------------------------------------------

    // Default: empty tensor
    Tensor() = default;

    // Construct a zero-filled tensor with a given shape.
    // e.g. Tensor({3, 4})  ->  12 zeros, shape = {3, 4}
    explicit Tensor(std::vector<int> shape_)
        : shape(std::move(shape_))
    {
        int total = num_elements();
        data.assign(total, 0.0f);
    }

    // Construct from existing data + shape (used in tests and weight loading).
    Tensor(std::vector<float> data_, std::vector<int> shape_)
        : data(std::move(data_)), shape(std::move(shape_))
    {
        assert((int)data.size() == num_elements() &&
               "data.size() does not match product of shape dims");
    }

    // --- Utility ------------------------------------------------------------

    // Total number of elements = product of all shape dimensions.
    int num_elements() const {
        if (shape.empty()) return 0;
        // std::accumulate multiplies all dims together, starting from 1.
        return std::accumulate(shape.begin(), shape.end(),
                               1, std::multiplies<int>());
    }

    // Number of dimensions (rank). A matrix has rank 2, a vector rank 1.
    int ndim() const { return (int)shape.size(); }

    // Print shape as a string, e.g. "[3, 4]"
    std::string shape_str() const {
        std::string s = "[";
        for (int i = 0; i < (int)shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < (int)shape.size()) s += ", ";
        }
        return s + "]";
    }

    // --- Element access (2D) ------------------------------------------------
    // For a shape {rows, cols}, element at (i, j) is at data[i*cols + j].
    // This is "row-major" order — same as C arrays, numpy default.

    float& at(int i, int j) {
        assert(ndim() == 2);
        return data[i * shape[1] + j];
    }
    float at(int i, int j) const {
        assert(ndim() == 2);
        return data[i * shape[1] + j];
    }

    // --- Element access (1D) ------------------------------------------------
    float& at(int i) {
        assert(ndim() == 1);
        return data[i];
    }
    float at(int i) const {
        assert(ndim() == 1);
        return data[i];
    }

    // Print contents to stdout (for debugging small tensors).
    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " " << shape_str() << ":\n";
        for (int i = 0; i < (int)data.size(); ++i) {
            std::cout << data[i];
            if (i + 1 < (int)data.size()) std::cout << ", ";
        }
        std::cout << "\n";
    }
};


// ---------------------------------------------------------------------------
// matmul(A, B)  ->  C = A @ B
// ---------------------------------------------------------------------------
// A has shape {M, K}
// B has shape {K, N}
// C has shape {M, N}
//
// The math: C[i][j] = sum over k of  A[i][k] * B[k][j]
//
// Why this matters for GPT-2:
//   Every linear projection (Q, K, V, output, FFN) is a matmul.
//   e.g. Q = x @ W_q   where x is {seq_len, d_model}, W_q is {d_model, d_head}
//
// Complexity: O(M * N * K)  — naive but correct and readable.
// ---------------------------------------------------------------------------
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2 &&
           "matmul expects 2D tensors");
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    assert(K == B.shape[0] &&
           "matmul: A columns must equal B rows");

    Tensor C({M, N}); // zero-initialised

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i][k] stored at data[i*K + k]
                // B[k][j] stored at data[k*N + j]
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
// matvec(A, b)  ->  y = A @ b
// ---------------------------------------------------------------------------
// A has shape {M, K}, b has shape {K}  ->  y has shape {M}
// Shorthand used when projecting a single token vector.
// ---------------------------------------------------------------------------
inline Tensor matvec(const Tensor& A, const Tensor& b) {
    assert(A.ndim() == 2 && b.ndim() == 1);
    int M = A.shape[0];
    int K = A.shape[1];
    assert(K == b.shape[0]);

    Tensor y({M});
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A.data[i * K + k] * b.data[k];
        y.data[i] = sum;
    }
    return y;
}

// ---------------------------------------------------------------------------
// add_bias(x, bias)  ->  x[i] += bias[i]  (in-place)
// ---------------------------------------------------------------------------
// Used after every linear projection: y = x @ W + b
// ---------------------------------------------------------------------------
inline void add_bias(Tensor& x, const Tensor& bias) {
    assert(x.ndim() == 1 && bias.ndim() == 1);
    assert(x.shape[0] == bias.shape[0]);
    for (int i = 0; i < x.shape[0]; ++i)
        x.data[i] += bias.data[i];
}

// ---------------------------------------------------------------------------
// softmax(x)  ->  probability distribution over x  (in-place)
// ---------------------------------------------------------------------------
// Formula:  softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// WHY subtract max(x)?
//   Numerical stability. exp(800) = inf in float32. By subtracting the max
//   first, the largest value becomes exp(0)=1, and all others are ≤ 1.
//   This does NOT change the result mathematically because:
//     exp(x[i] - c) / Σ exp(x[j] - c)  =  exp(x[i]) / Σ exp(x[j])
//   (the c cancels in numerator and denominator)
//
// WHY this matters for attention:
//   The attention scores (Q @ K^T / sqrt(d_k)) can be large. Softmax turns
//   them into weights that sum to 1, so the output is a weighted average of V.
// ---------------------------------------------------------------------------
inline void softmax(Tensor& x) {
    assert(x.ndim() == 1);
    int N = x.shape[0];

    // Step 1: find the max value for numerical stability
    float max_val = x.data[0];
    for (int i = 1; i < N; ++i)
        if (x.data[i] > max_val) max_val = x.data[i];

    // Step 2: subtract max, then exponentiate
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        x.data[i] = std::exp(x.data[i] - max_val);
        sum += x.data[i];
    }

    // Step 3: divide by sum -> values now sum to 1.0
    for (int i = 0; i < N; ++i)
        x.data[i] /= sum;
}

// ---------------------------------------------------------------------------
// layernorm(x, gamma, beta, eps)  ->  normalised x  (in-place)
// ---------------------------------------------------------------------------
// Formula:  y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps)  +  beta[i]
//
// Steps:
//   1. Compute mean of all elements in x
//   2. Compute variance
//   3. Normalise: subtract mean, divide by std-dev
//   4. Scale & shift with learned parameters gamma (weight) and beta (bias)
//
// WHY it helps:
//   Deep networks suffer from "internal covariate shift" — the distribution
//   of each layer's input keeps changing during training, making learning
//   unstable. LayerNorm re-centres and re-scales the activations back to
//   a predictable range after every sub-layer.
//
// GPT-2 specifics:
//   Uses "Pre-LN" — LayerNorm is applied BEFORE attention and FFN,
//   not after (unlike the original Transformer). This makes training more
//   stable.  Shape: x is {d_model}, gamma and beta are also {d_model}.
// ---------------------------------------------------------------------------
inline void layernorm(Tensor& x, const Tensor& gamma, const Tensor& beta,
                      float eps = 1e-5f) {
    assert(x.ndim() == 1);
    assert(gamma.shape == x.shape && beta.shape == x.shape);

    int N = x.shape[0];

    // Step 1: mean
    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += x.data[i];
    mean /= N;

    // Step 2: variance  E[(x - mean)^2]
    float var = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = x.data[i] - mean;
        var += diff * diff;
    }
    var /= N;

    // Step 3 + 4: normalise, then scale & shift
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < N; ++i) {
        float x_norm = (x.data[i] - mean) * inv_std;
        x.data[i] = gamma.data[i] * x_norm + beta.data[i];
    }
}

// ---------------------------------------------------------------------------
// gelu(x)  ->  Gaussian Error Linear Unit  (in-place)
// ---------------------------------------------------------------------------
// Approximate formula used by GPT-2:
//   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// WHY not just ReLU?
//   ReLU(x) = max(0, x) — hard gate: fully blocks negative inputs.
//   GELU is a smooth, probabilistic gate: it scales x by how likely it is
//   that a Gaussian N(0,1) is ≤ x. This means small negative values are
//   only *partially* suppressed, not zeroed out. Empirically this leads to
//   better gradients and model quality for language tasks.
//
// The tanh approximation is numerically fast and matches the exact GELU
// to within floating-point noise. OpenAI used this exact formula in GPT-2.
// ---------------------------------------------------------------------------
inline void gelu(Tensor& x) {
    assert(x.ndim() == 1);
    // Precompute constant: sqrt(2 / pi) ≈ 0.7978845608
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    for (int i = 0; i < (int)x.data.size(); ++i) {
        float v = x.data[i];
        // inner = sqrt(2/π) * (x + 0.044715 * x^3)
        float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        // GELU(x) = 0.5 * x * (1 + tanh(inner))
        x.data[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}