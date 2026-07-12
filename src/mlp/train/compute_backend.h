#pragma once

#include <cstddef>
#include <memory>

// The training compute abstraction (design.ja.md 5.9, mlp_module_design.ja.md
// 3.1): matrix products, activations and elementwise ops behind a virtual
// interface, so CPU (BLAS) / CUDA / Metal implementations can be swapped at
// startup. Training is fp32; virtual dispatch is irrelevant next to the GEMM
// cost. Inference never touches this (int16 hand-written kernels, 5.6).
//
// Embedding gather/scatter is currently plain loops at the call sites
// (dataset.cpp gather, net.cpp scatter): they are memory-bound and BLAS has
// nothing to offer. A GPU backend will need them added here so the data can
// stay on-device.

namespace segmentlib::mlp::train {

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Row-major GEMM: C (m×n) = alpha · op(A) (m×k) · op(B) (k×n) + beta · C,
    // where op transposes when the corresponding flag is set. Leading
    // dimensions are those of the *stored* matrices, as in CBLAS.
    virtual void gemm(bool trans_a, bool trans_b, std::size_t m, std::size_t n,
                      std::size_t k, float alpha, const float* a,
                      std::size_t lda, const float* b, std::size_t ldb,
                      float beta, float* c, std::size_t ldc) = 0;

    // h[i] = max(a[i], 0)
    virtual void relu(const float* a, float* h, std::size_t n) = 0;

    // ReLU': dh[i] = a[i] > 0 ? dh[i] : 0 (masks the upstream gradient in
    // place, given the pre-activation `a`).
    virtual void relu_backward(const float* a, float* dh, std::size_t n) = 0;

    // y[i] += alpha * x[i]
    virtual void axpy(std::size_t n, float alpha, const float* x, float* y) = 0;
};

// The BLAS-backed CPU implementation (Accelerate / OpenBLAS / MKL — chosen at
// link time by SEGMENTLIB_BLAS).
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();

}  // namespace segmentlib::mlp::train
