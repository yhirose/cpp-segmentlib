#include "mlp/train/compute_backend.h"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace segmentlib::mlp::train {

namespace {

class CpuBlasBackend final : public ComputeBackend {
public:
    void gemm(bool trans_a, bool trans_b, std::size_t m, std::size_t n,
              std::size_t k, float alpha, const float* a, std::size_t lda,
              const float* b, std::size_t ldb, float beta, float* c,
              std::size_t ldc) override {
        if (m == 0 || n == 0) {
            return;
        }
        cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
                    trans_b ? CblasTrans : CblasNoTrans, static_cast<int>(m),
                    static_cast<int>(n), static_cast<int>(k), alpha, a,
                    static_cast<int>(lda), b, static_cast<int>(ldb), beta, c,
                    static_cast<int>(ldc));
    }

    void relu(const float* a, float* h, std::size_t n) override {
        for (std::size_t i = 0; i < n; ++i) {
            h[i] = a[i] > 0.0f ? a[i] : 0.0f;
        }
    }

    void relu_backward(const float* a, float* dh, std::size_t n) override {
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] <= 0.0f) {
                dh[i] = 0.0f;
            }
        }
    }

    void axpy(std::size_t n, float alpha, const float* x, float* y) override {
        cblas_saxpy(static_cast<int>(n), alpha, x, 1, y, 1);
    }
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend() {
    return std::make_unique<CpuBlasBackend>();
}

}  // namespace segmentlib::mlp::train
