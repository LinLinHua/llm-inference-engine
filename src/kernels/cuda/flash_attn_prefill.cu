/*
src/kernels/cuda/flash_attn_prefill.cu

FlashAttention-2 Prefill Kernel (CUDA, Tensor Core, GQA)
=========================================================
Uses dynamic shared memory to support BR=BC=64 tile size on A100.
cudaFuncSetAttribute raises the shared memory limit to 164KB.

Note: torch/extension.h must NOT be included in .cu files.
      Use ATen/ATen.h instead. Python bindings go in the .cpp file.
      Reference: https://pytorch.org/docs/stable/cpp_extension.html
*/

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define BR     64
#define BC     64
#define HD    128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// shared memory layout (dynamic):
// Q_smem: BR×HD fp16  = 64×128×2 = 16KB
// K_smem: BC×HD fp16  = 16KB
// V_smem: BC×HD fp16  = 16KB
// S_smem: BR×BC fp32  = 64×64×4 = 16KB
// total: 64KB < 164KB A100 limit

__global__ void fa2_prefill_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
          __half* __restrict__ O,
    int B, int Hq, int Hkv, int Sq, int Skv,
    float scale, bool causal
) {
    int q_block = blockIdx.x;
    int h       = blockIdx.y;
    int b       = blockIdx.z;
    int kv_h    = h / (Hq / Hkv);

    int warp_id = threadIdx.x / 32;  // 0..3
    int lane_id = threadIdx.x % 32;
    int tid     = threadIdx.x;

    int q_start = q_block * BR;
    if (q_start >= Sq) return;

    // Dynamic shared memory
    extern __shared__ char smem_buf[];
    __half* Q_smem = (__half*)smem_buf;
    __half* K_smem = Q_smem + BR * HD;
    __half* V_smem = K_smem + BC * HD;
    float*  S_smem = (float*)(V_smem + BC * HD);

    // Load Q tile
    for (int idx = tid; idx < BR * HD; idx += blockDim.x) {
        int r  = idx / HD;
        int c  = idx % HD;
        int gq = q_start + r;
        Q_smem[r * HD + c] = (gq < Sq) ?
            Q[b * Hq * Sq * HD + h * Sq * HD + gq * HD + c] :
            __float2half(0.f);
    }
    __syncthreads();

    int warp_row_start = warp_id * WMMA_M;  // 0, 16, 32, 48

    float m_i[WMMA_M];
    float l_i[WMMA_M];
    float O_acc[WMMA_M][4];

    for (int r = 0; r < WMMA_M; r++) {
        m_i[r] = -FLT_MAX;
        l_i[r] = 0.f;
        for (int d = 0; d < 4; d++) O_acc[r][d] = 0.f;
    }

    int num_kv_blocks = (Skv + BC - 1) / BC;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BC;

        if (causal && kv_start > q_start + BR - 1) break;

        // Load K tile
        for (int idx = tid; idx < BC * HD; idx += blockDim.x) {
            int r  = idx / HD;
            int c  = idx % HD;
            int gk = kv_start + r;
            K_smem[r * HD + c] = (gk < Skv) ?
                K[b * Hkv * Skv * HD + kv_h * Skv * HD + gk * HD + c] :
                __float2half(0.f);
        }

        // Load V tile
        for (int idx = tid; idx < BC * HD; idx += blockDim.x) {
            int r  = idx / HD;
            int c  = idx % HD;
            int gk = kv_start + r;
            V_smem[r * HD + c] = (gk < Skv) ?
                V[b * Hkv * Skv * HD + kv_h * Skv * HD + gk * HD + c] :
                __float2half(0.f);
        }
        __syncthreads();

        // Tensor Core GEMM: S[BR,BC] = Q_smem @ K_smem^T
        for (int nc = 0; nc < BC / WMMA_N; nc++) {
            wmma::fragment<wmma::accumulator,
                WMMA_M, WMMA_N, WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.f);

            for (int dk = 0; dk < HD; dk += WMMA_K) {
                wmma::fragment<wmma::matrix_a,
                    WMMA_M, WMMA_N, WMMA_K, __half,
                    wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b,
                    WMMA_M, WMMA_N, WMMA_K, __half,
                    wmma::col_major> k_frag;

                wmma::load_matrix_sync(q_frag,
                    &Q_smem[warp_row_start * HD + dk], HD);
                wmma::load_matrix_sync(k_frag,
                    &K_smem[nc * WMMA_N * HD + dk], BC);
            wmma::mma_sync(acc, q_frag, k_frag, acc);
            }

            wmma::store_matrix_sync(
                &S_smem[warp_row_start * BC + nc * WMMA_N],
                acc, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax + O_acc update
        for (int r = 0; r < WMMA_M; r++) {
            int global_i = q_start + warp_row_start + r;
            if (global_i >= Sq) continue;

            float m_tile = -FLT_MAX;
            for (int jj = 0; jj < BC; jj++) {
                int global_j = kv_start + jj;
                float s = S_smem[(warp_row_start + r) * BC + jj] * scale;
                if (causal && global_j > global_i) s = -1e9f;
                if (global_j >= Skv)               s = -1e9f;
                S_smem[(warp_row_start + r) * BC + jj] = s;
                m_tile = fmaxf(m_tile, s);
            }

            float m_new   = fmaxf(m_i[r], m_tile);
            float rescale = expf(m_i[r] - m_new);
            float l_tile  = 0.f;

            for (int jj = 0; jj < BC; jj++) {
                S_smem[(warp_row_start + r) * BC + jj] =
                    expf(S_smem[(warp_row_start + r) * BC + jj] - m_new);
                l_tile += S_smem[(warp_row_start + r) * BC + jj];
            }

            for (int di = 0; di < 4; di++) {
                int d = lane_id + di * 32;
                float v_acc = 0.f;
                for (int jj = 0; jj < BC; jj++)
                    v_acc += S_smem[(warp_row_start + r) * BC + jj] *
                             __half2float(V_smem[jj * HD + d]);
                O_acc[r][di] = rescale * O_acc[r][di] + v_acc;
            }

            l_i[r] = rescale * l_i[r] + l_tile;
            m_i[r] = m_new;
        }
        __syncthreads();
    }

    // Write output to HBM
    for (int r = 0; r < WMMA_M; r++) {
        int global_i = q_start + warp_row_start + r;
        if (global_i >= Sq || l_i[r] == 0.f) continue;

        for (int di = 0; di < 4; di++) {
            int d = lane_id + di * 32;
            O[b * Hq * Sq * HD + h * Sq * HD + global_i * HD + d] =
                __float2half(O_acc[r][di] / l_i[r]);
        }
    }
}

// ── C++ wrapper ────────────────────────────────────────────────────────────
at::Tensor fa2_prefill(
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == at::kHalf);
    int B   = Q.size(0);
    int Hq  = Q.size(1);
    int Sq  = Q.size(2);
    int D   = Q.size(3);
    int Hkv = K.size(1);
    int Skv = K.size(2);
    TORCH_CHECK(D == 128, "head_dim must be 128");

    auto O      = at::zeros_like(Q);
    float scale = 1.f / sqrtf((float)D);

    size_t smem_size = (size_t)(BR + BC + BC) * HD * sizeof(__half)
                     + (size_t)BR * BC * sizeof(float);

    cudaFuncSetAttribute(
        fa2_prefill_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    dim3 grid((Sq + BR - 1) / BR, Hq, B);
    dim3 block(128);  // 4 warps × 32 threads

    fa2_prefill_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(K.data_ptr()),
        reinterpret_cast<const __half*>(V.data_ptr()),
        reinterpret_cast<      __half*>(O.data_ptr()),
        B, Hq, Hkv, Sq, Skv, scale, causal
    );
    return O;
}