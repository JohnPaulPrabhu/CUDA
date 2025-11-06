#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using std::cout;
using std::endl;

// ------------------------------------------------------------
// Helpers: CUDA / cuBLAS error checking
// ------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                               \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",            \
                         __FILE__, __LINE__, int(_s));                         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ------------------------------------------------------------
// Naive CUDA kernel for matrix multiplication
// A: (m x k), B: (k x n), C: (m x n)  [row-major]
// ------------------------------------------------------------
__global__ void matmul_gpu(const float* A, const float* B, float* C,
                           int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ------------------------------------------------------------
// CPU reference implementation (row-major)
// ------------------------------------------------------------
void matmul_cpu(const float* A, const float* B, float* C,
                int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += A[i * k + t] * B[t * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// cuBLAS SGEMM wrapper for row-major inputs
//
// Note: cuBLAS assumes column-major matrices. For row-major A(m,k),
// B(k,n), C(m,n) we can compute C = A * B by swapping A/B and using
// their row-major leading dimensions as "lda/ldb/ldc" with opN,
// effectively computing C^T = B^T * A^T in column-major.
// This is the common "row-major trick" used below.
// ------------------------------------------------------------
// Create one global-ish handle you reuse
cublasHandle_t g_handle;
void gpu_cublas_gemm(const float* d_A, const float* d_B, float* d_C,
                     int m, int k, int n) {

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Column-major SGEMM signature:
    // C = alpha * op(A) * op(B) + beta * C
    // Using row-major data, this call computes C correctly:
    // Treat row-major A(m,k) as column-major A^T(k,m), etc.
    cublasSgemm(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,      // op(B), op(A)
        n,                             // rows of op(B) and C^T
        m,                             // cols of op(A) and C^T
        k,                             // shared dim
        &alpha,
        d_B, n,                        // B, ldb
        d_A, k,                        // A, lda
        &beta,
        d_C, n);                      // C (row-major) as C^T in column-major
}

int main() {
    // Problem size
    // const int m = 512, k = 512, n = 512;
    const int m = 90, k = 90, n = 90;

    // Host-side array
    float* h_A;
    int size = m * k;
    h_A = (float*)malloc(size * sizeof(float));
    // Initialize h_array with values
    for (int i = 0; i < size; ++i) {
        h_A[i] = i;
    }

    // Host-side array
    float* h_B;
    // int size = m * k;
    h_B = (float*)malloc(size * sizeof(float));
    // Initialize h_array with values
    for (int i = size; i > 0; --i) {
        h_B[i] = i;
    }


    float h_C[m * n]   = {0}; // kernel result
    float h_C2[m * n]  = {0}; // cuBLAS result
    float h_C_CPU[m * n] = {0}; // CPU reference

    // Create handle once
    CUBLAS_CHECK(cublasCreate(&g_handle));

    // Device buffers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C2, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Run GPU kernel
    matmul_gpu<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    const int iters = 10000; // small matrices need many iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        matmul_gpu<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("Kernel avg: %.3f microseconds\n", (ms * 1000.0f) / iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU timing
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        matmul_cpu(h_A, h_B, h_C_CPU, m, k, n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("CPU avg: %.3f microseconds\n", us / iters);

    // cuBLAS timing
    gpu_cublas_gemm(d_A, d_B, d_C2, m, k, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        gpu_cublas_gemm(d_A, d_B, d_C2, m, k, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("cuBLAS avg: %.3f microseconds\n", (ms * 1000.0f) / iters);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy kernel output back
    CUDA_CHECK(cudaMemcpy(h_C,  d_C,  m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C2, d_C2, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Show results
    cout << "Result from kernel (C = A x B):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) cout << h_C[i * n + j] << " ";
        cout << endl;
    }

    cout << "Result from cuBLAS (C2 = A x B):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) cout << h_C2[i * n + j] << " ";
        cout << endl;
    }

    cout << "Result from CPU reference:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) cout << h_C_CPU[i * n + j] << " ";
        cout << endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C2));
    
    // Destroy handle once
    CUBLAS_CHECK(cublasDestroy(g_handle));

    return 0;
}