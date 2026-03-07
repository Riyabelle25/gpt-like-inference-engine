#pragma once
// tensor.h - Tensor class + math primitives for GPT-2 inference.
//
// Milestone 4 adds AVX2+FMA SIMD acceleration to the two most expensive ops:
//   • matvec  - called ~100× per decode token (every linear projection)
//   • matmul  - called during prefill attention (seq_len × seq_len × d_head)
//
// The file is organised as follows:
//   1. Tensor struct         - flat float storage + helpers
//   2. matmul / matvec       - SIMD-accelerated, scalar fallback
//   3. add_bias / softmax    - unchanged
//   4. layernorm / gelu      - AVX2-accelerated horizontal reductions
//
// ===========================================================================
// WHY AVX2 + FMA?
// ===========================================================================
// A modern CPU can do 8 float multiplications per clock cycle per core using
// AVX2 (256-bit SIMD = 8 × float32).  Add FMA (fused multiply-add) and you
// get 8 multiply-adds per clock - twice the throughput of separate mul+add.
//
// For a matvec with M=2304, K=768 (c_attn: projects x to Q+K+V together):
//   Scalar:  2304 × 768 = 1,769,472 scalar multiply-adds
//   AVX2/FMA: the inner K-loop processes 8 elements at once -> up to 8× faster
//             (real speedup is memory-bandwidth-limited, typically 3–5×)
//
// Key intrinsics used:
//   _mm256_loadu_ps(ptr)          - load 8 floats (unaligned OK)
//   _mm256_fmadd_ps(a, b, acc)    - acc += a * b  (fused, one instruction)
//   _mm256_add_ps(a, b)           - element-wise add
//   _mm256_hadd_ps(a, b)          - horizontal add pairs within 128-bit halves
//   _mm256_extractf128_ps(v, 1)   - extract upper 128-bit lane
//
// Compile with: -mavx2 -mfma 
// ===========================================================================

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <numeric>
#include <functional>
#include <cassert>
#include <cstdint>

// Conditionally include AVX2 intrinsics.
// Falls back gracefully to scalar if AVX2/FMA are not available.
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define USE_AVX2 1
#else
  #define USE_AVX2 0
#endif

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
struct Tensor {
    std::vector<float> data;
    std::vector<int>   shape;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_)
        : shape(std::move(shape_))
    {
        data.assign(num_elements(), 0.0f);
    }

    Tensor(std::vector<float> data_, std::vector<int> shape_)
        : data(std::move(data_)), shape(std::move(shape_))
    {
        assert((int)data.size() == num_elements());
    }

    int num_elements() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    int ndim() const { return (int)shape.size(); }

    std::string shape_str() const {
        std::string s = "[";
        for (int i = 0; i < (int)shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i+1 < (int)shape.size()) s += ", ";
        }
        return s + "]";
    }

    float& at(int i, int j) { assert(ndim()==2); return data[i*shape[1]+j]; }
    float  at(int i, int j) const { assert(ndim()==2); return data[i*shape[1]+j]; }
    float& at(int i)        { assert(ndim()==1); return data[i]; }
    float  at(int i)  const { assert(ndim()==1); return data[i]; }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " " << shape_str() << ":\n";
        for (int i = 0; i < (int)data.size(); ++i) {
            std::cout << data[i];
            if (i+1 < (int)data.size()) std::cout << ", ";
        }
        std::cout << "\n";
    }
};


// ===========================================================================
// SIMD helpers (compiled away to nothing if USE_AVX2 == 0)
// ===========================================================================

#if USE_AVX2

// ---------------------------------------------------------------------------
// hsum256(v) - reduce an 8-wide AVX register to a single float sum
// ---------------------------------------------------------------------------
// AVX has no single "add all 8 lanes" instruction, so we reduce in two
// rounds of _mm256_hadd_ps (horizontal add), then combine the two 128-bit
// halves:
//
//   Input: [a0 a1 a2 a3 | a4 a5 a6 a7]   (| = lane boundary)
//   hadd:  [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
//   hadd:  [sum0123, sum0123, ... | sum4567, sum4567, ...]
//   add upper + lower 128-bit halves -> [sum0123+sum4567, ...]
//   extract lane 0 -> grand total
// ---------------------------------------------------------------------------
inline float hsum256(__m256 v) {
    __m256 h1  = _mm256_hadd_ps(v,  v);
    __m256 h2  = _mm256_hadd_ps(h1, h1);
    __m128 hi  = _mm256_extractf128_ps(h2, 1);
    __m128 lo  = _mm256_castps256_ps128(h2);
    return _mm_cvtss_f32(_mm_add_ps(lo, hi));
}

// ---------------------------------------------------------------------------
// dot_avx(a, b, K) - dot product of two float arrays of length K
// ---------------------------------------------------------------------------
// Unrolled to 2 accumulators to hide FMA latency (FMA has ~5 cycle latency
// on Skylake; using 2 independent accumulators keeps 2 FMAs in-flight).
// Tail (K % 8 remainder) handled scalar.
// ---------------------------------------------------------------------------
inline float dot_avx(const float* __restrict__ a,
                     const float* __restrict__ b,
                     int K) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int k = 0;
    // 2× unrolled: 16 floats per iteration, 2 independent FMA chains
    for (; k + 16 <= K; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+k),   _mm256_loadu_ps(b+k),   acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+k+8), _mm256_loadu_ps(b+k+8), acc1);
    }
    // Merge the two accumulators
    float result = hsum256(_mm256_add_ps(acc0, acc1));
    // Handle remaining 8-wide chunk (if K%16 >= 8)
    if (k + 8 <= K) {
        result += hsum256(_mm256_mul_ps(_mm256_loadu_ps(a+k), _mm256_loadu_ps(b+k)));
        k += 8;
    }
    // Scalar tail (< 8 remaining)
    for (; k < K; ++k) result += a[k] * b[k];
    return result;
}

#endif // USE_AVX2


// ===========================================================================
// matvec(A, b) -> y = A @ b
// ===========================================================================
// A: {M, K},  b: {K}  ->  y: {M}
//
// y[i] = dot(A_row_i, b)  for each i in [0, M)
//
// WHY this is the hottest function in GPT-2 decode:
//   Every linear projection in every layer is a matvec call with a single
//   token vector (shape {K}) against a weight matrix.  Per token, we call
//   this function roughly 4 times per block × 12 blocks = 48+ times.
//
// SIMD strategy:
//   Each row of A is contiguous in memory -> dot_avx processes 16 floats/iter.
//   With K=768 (d_model) that's 48 AVX iterations per output element.
// ===========================================================================
inline Tensor matvec(const Tensor& A, const Tensor& b) {
    assert(A.ndim() == 2 && b.ndim() == 1);
    int M = A.shape[0];
    int K = A.shape[1];
    assert(K == b.shape[0]);

    Tensor y({M});

#if USE_AVX2
    const float* bptr = b.data.data();
    for (int i = 0; i < M; ++i)
        y.data[i] = dot_avx(A.data.data() + (size_t)i * K, bptr, K);
#else
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A.data[i * K + k] * b.data[k];
        y.data[i] = sum;
    }
#endif
    return y;
}


// ===========================================================================
// matmul(A, B) -> C = A @ B
// ===========================================================================
// A: {M, K},  B: {K, N}  ->  C: {M, N}
//
// Only called during prefill (M = prompt length > 1).
// Decode path only calls matvec (M = 1 always).
//
// SIMD strategy - transpose-then-dot:
//   Naively C[i][j] = dot(A_row_i, B_col_j), but B columns are strided
//   ({stride = N} between elements), which kills cache performance.
//   Instead:
//     1. Transpose B -> Bt {N, K} so each column becomes a contiguous row.
//     2. C[i][j] = dot(A_row_i, Bt_row_j) - both operands are contiguous.
//     3. Use dot_avx for each pair.
//
//   Transpose cost: O(K*N) = O(768*768) = 589,824 writes - paid once per call,
//   worth it when M ≥ 2 because we save M × N × K/8 AVX load strides.
// ===========================================================================
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    assert(K == B.shape[0]);

    Tensor C({M, N});

#if USE_AVX2
    // Step 1: Transpose B -> Bt[n][k] = B[k][n]
    std::vector<float> Bt((size_t)N * K);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            Bt[(size_t)n * K + k] = B.data[(size_t)k * N + n];

    // Step 2: C[i][j] = dot(A_row_i, Bt_row_j) using AVX
    for (int i = 0; i < M; ++i) {
        const float* a_row = A.data.data() + (size_t)i * K;
        for (int j = 0; j < N; ++j)
            C.data[(size_t)i * N + j] =
                dot_avx(a_row, Bt.data() + (size_t)j * K, K);
    }
#else
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A.data[i * K + k] * B.data[k * N + j];
            C.data[i * N + j] = sum;
        }
#endif
    return C;
}


// ---------------------------------------------------------------------------
// add_bias(x, bias) - x[i] += bias[i]  (in-place)
// ---------------------------------------------------------------------------
inline void add_bias(Tensor& x, const Tensor& bias) {
    assert(x.ndim() == 1 && bias.ndim() == 1);
    assert(x.shape[0] == bias.shape[0]);
    for (int i = 0; i < x.shape[0]; ++i)
        x.data[i] += bias.data[i];
}


// ---------------------------------------------------------------------------
// softmax(x) - numerically stable softmax  (in-place)
// ---------------------------------------------------------------------------
inline void softmax(Tensor& x) {
    assert(x.ndim() == 1);
    int N = x.shape[0];
    float max_val = x.data[0];
    for (int i = 1; i < N; ++i)
        if (x.data[i] > max_val) max_val = x.data[i];
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) { x.data[i] = std::exp(x.data[i] - max_val); sum += x.data[i]; }
    for (int i = 0; i < N; ++i) x.data[i] /= sum;
}


// ---------------------------------------------------------------------------
// layernorm(x, gamma, beta, eps) - Pre-LN normalisation  (in-place)
// ---------------------------------------------------------------------------
// Formula:  y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
//
// AVX2 accelerates the two linear passes (mean, variance) over the 768
// element vector.  The final scale+shift remains scalar for clarity.
// ---------------------------------------------------------------------------
inline void layernorm(Tensor& x, const Tensor& gamma, const Tensor& beta,
                      float eps = 1e-5f) {
    assert(x.ndim() == 1);
    assert(gamma.shape == x.shape && beta.shape == x.shape);
    int N = x.shape[0];
    const float* xp = x.data.data();

    // --- Mean ---
    float mean = 0.0f;
#if USE_AVX2
    {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= N; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(xp+i));
        mean = hsum256(acc);
        for (; i < N; ++i) mean += xp[i];
    }
#else
    for (int i = 0; i < N; ++i) mean += xp[i];
#endif
    mean /= N;

    // --- Variance E[(x-mean)^2] ---
    float var = 0.0f;
#if USE_AVX2
    {
        __m256 vacc  = _mm256_setzero_ps();
        __m256 vmean = _mm256_set1_ps(mean);
        int i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(xp+i), vmean);
            vacc = _mm256_fmadd_ps(d, d, vacc);
        }
        var = hsum256(vacc);
        for (; i < N; ++i) { float d = xp[i]-mean; var += d*d; }
    }
#else
    for (int i = 0; i < N; ++i) { float d = xp[i]-mean; var += d*d; }
#endif
    var /= N;

    // --- Normalise + scale + shift ---
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < N; ++i) {
        float xn = (xp[i] - mean) * inv_std;
        x.data[i] = gamma.data[i] * xn + beta.data[i];
    }
}


// ---------------------------------------------------------------------------
// gelu(x) - GELU activation  (in-place)
// ---------------------------------------------------------------------------
// tanh is transcendental; no AVX2 intrinsic. Scalar with -O2 auto-vec.
// ---------------------------------------------------------------------------
inline void gelu(Tensor& x) {
    assert(x.ndim() == 1);
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    for (int i = 0; i < (int)x.data.size(); ++i) {
        float v = x.data[i];
        float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        x.data[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}