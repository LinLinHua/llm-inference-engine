/*
FlashAttention-2 Prefill Kernel (CUDA, Tensor Core, GQA)
=========================================================
Used during prefill phase: Q, K, V all have seq_len = prompt length.

Key optimizations vs naive attention:
  1. Online softmax     → no N×N matrix written to HBM
  2. SRAM tiling        → Q/K/V tiles fit in shared memory
  3. Tensor Core MMA    → wmma fp16 → fp32, much faster than CUDA cores
  4. FA-2 loop order    → outer=Q blocks, inner=KV blocks
                          each thread block owns one Q tile
                          O_acc stays in register, written to HBM only once
  5. GQA support        → kv_head = q_head / group_size

HBM traffic: O(N×D) vs O(N²) for naive attention
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <float.h>

using namespace nvcuda;

#define BR     64    // Q tile rows
#define BC     64    // KV tile rows  
#define HD    128    // head_dim (Llama-3.x = 128)

// Tensor Core tile size (fixed by hardware)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void fa2_prefill_kernel(
    const __half* __restrict__ Q,    // [B, Hq,  Sq,  D]
    const __half* __restrict__ K,    // [B, Hkv, Skv, D]
    const __half* __restrict__ V,    // [B, Hkv, Skv, D]
          __half* __restrict__ O,    // [B, Hq,  Sq,  D]
    int B, int Hq, int Hkv, int Sq, int Skv,
    float scale, bool causal
) {
    // FA-2: each thread block owns one Q tile
    int q_block = blockIdx.x;
    int h       = blockIdx.y;
    int b       = blockIdx.z;
    int kv_h    = h / (Hq / Hkv);   // GQA mapping

    // Thread indices
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid     = threadIdx.x;

    int q_start = q_block * BR;
    if (q_start >= Sq) return;

    // ── Shared memory ──────────────────────────────────────────────────────
    __shared__ __half Q_smem[BR][HD];   // 64×128 fp16 = 16KB
    __shared__ __half K_smem[BC][HD];   // 16KB
    __shared__ __half V_smem[BC][HD];   // 16KB
    __shared__ float  S_smem[BR][BC];   // 64×64 fp32 = 16KB

    // ── Load Q tile from HBM to shared memory ─────────────────────────────
    for (int idx = tid; idx < BR * HD; idx += blockDim.x) {
        int r  = idx / HD;
        int c  = idx % HD;
        int gq = q_start + r;
        Q_smem[r][c] = (gq < Sq) ?
            Q[b * Hq * Sq * HD + h * Sq * HD + gq * HD + c] :
            __float2half(0.f);
    }
    __syncthreads();

    // ── Per-warp softmax state (lives in register) ─────────────────────────
    int warp_row_start = warp_id * WMMA_M;  // 0, 16, 32, 48

    float m_i[WMMA_M];       // running max
    float l_i[WMMA_M];       // running sum of exp
    float O_acc[WMMA_M][4];  // output accumulator (each thread owns 4 cols)

    for (int r = 0; r < WMMA_M; r++) {
        m_i[r] = -FLT_MAX;
        l_i[r] = 0.f;
        for (int d = 0; d < 4; d++) O_acc[r][d] = 0.f;
    }

    // ── Iterate over KV blocks (FA-2 inner loop) ───────────────────────────
    int num_kv_blocks = (Skv + BC - 1) / BC;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BC;

        // Causal skip: if entire KV tile is beyond current Q position, stop
        if (causal && kv_start > q_start + BR - 1) break;

        // Load K tile
        for (int idx = tid; idx < BC * HD; idx += blockDim.x) {
            int r  = idx / HD;
            int c  = idx % HD;
            int gk = kv_start + r;
            K_smem[r][c] = (gk < Skv) ?
                K[b * Hkv * Skv * HD + kv_h * Skv * HD + gk * HD + c] :
                __float2half(0.f);
        }

        // Load V tile
        for (int idx = tid; idx < BC * HD; idx += blockDim.x) {
            int r  = idx / HD;
            int c  = idx % HD;
            int gk = kv_start + r;
            V_smem[r][c] = (gk < Skv) ?
                V[b * Hkv * Skv * HD + kv_h * Skv * HD + gk * HD + c] :
                __float2half(0.f);
        }
        __syncthreads();

        // ── Tensor Core GEMM: S[BR, BC] = Q_smem @ K_smem^T ──────────────
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
                    &Q_smem[warp_row_start][dk], HD);
                wmma::load_matrix_sync(k_frag,
                    &K_smem[nc * WMMA_N][dk], HD);
                wmma::mma_sync(acc, q_frag, k_frag, acc);
            }

            wmma::store_matrix_sync(
                &S_smem[warp_row_start][nc * WMMA_N],
                acc, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // ── Scale + causal mask + online softmax update ────────────────────
        for (int r = 0; r < WMMA_M; r++) {
            int global_i = q_start + warp_row_start + r;
            if (global_i >= Sq) continue;

            // Find tile max
            float m_tile = -FLT_MAX;
            for (int jj = 0; jj < BC; jj++) {
                int global_j = kv_start + jj;
                float s = S_smem[warp_row_start + r][jj] * scale;
                if (causal && global_j > global_i) s = -1e9f;
                if (global_j >= Skv)               s = -1e9f;
                S_smem[warp_row_start + r][jj] = s;
                m_tile = fmaxf(m_tile, s);
            }

            // Online softmax recurrence
            float m_new   = fmaxf(m_i[r], m_tile);
            float rescale = expf(m_i[r] - m_new);
            float l_tile  = 0.f;

            for (int jj = 0; jj < BC; jj++) {
                S_smem[warp_row_start + r][jj] =
                    expf(S_smem[warp_row_start + r][jj] - m_new);
                l_tile += S_smem[warp_row_start + r][jj];
            }

            // Update O_acc: each thread handles 4 columns
            for (int di = 0; di < 4; di++) {
                int d = lane_id + di * 32;
                float v_acc = 0.f;
                for (int jj = 0; jj < BC; jj++)
                    v_acc += S_smem[warp_row_start + r][jj] *
                             __half2float(V_smem[jj][d]);
                O_acc[r][di] = rescale * O_acc[r][di] + v_acc;
            }

            l_i[r] = rescale * l_i[r] + l_tile;
            m_i[r] = m_new;
        }
        __syncthreads();
    }  // end KV loop

    // ── Write output to HBM ────────────────────────────────────────────────
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

// ── PyTorch binding ────────────────────────────────────────────────────────
torch::Tensor fa2_prefill(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat16);
    int B   = Q.size(0);
    int Hq  = Q.size(1);
    int Sq  = Q.size(2);
    int D   = Q.size(3);
    int Hkv = K.size(1);
    int Skv = K.size(2);
    TORCH_CHECK(D == 128, "head_dim must be 128");

    auto O      = torch::zeros_like(Q);
    float scale = 1.f / sqrtf((float)D);

    dim3 grid((Sq + BR - 1) / BR, Hq, B);
    dim3 block(128);  // 4 warps × 32 threads

    fa2_prefill_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(K.data_ptr()),
        reinterpret_cast<const __half*>(V.data_ptr()),
        reinterpret_cast<      __half*>(O.data_ptr()),
        B, Hq, Hkv, Sq, Skv, scale, causal
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa2_prefill", &fa2_prefill,
          "FlashAttention-2 prefill kernel (CUDA, Tensor Core, GQA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("causal") = true);
}