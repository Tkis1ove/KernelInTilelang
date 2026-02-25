#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstdint>
#include "ptx.cuh"

/*
q           :[b, l_q, d]
k, v        :[b, l_kv, d]
rel1, rel2  :[l_q, l_kv, d]
mask        :[b, l_q, l_kv]

sim1 = q @ k / sqrt(d)           :[b, l_q, l_kv]
sim2 = q @ rel1 / sqrt(d)        :[b, l_q, l_kv]
s = softmax(sim1 + sim2) + mask  :[b, l_q, l_kv]

out1 = s @ v     :[b, l_q, d]
out2 = s @ rel2  :[b, l_q, d]

o = out1 + out2  :[b, l_q, d]

l_q = l_kv = 16
d = 64
*/

#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__global__ void fusion_kernel(const half* q, const half* k, 
                              const half* v, half* o,
                              const half* rel1, const half* rel2, 
                              const half* mask,
                              const int b, const int l_q, const int l_kv, const int d) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;

    float scale = rsqrtf((float)d);

    constexpr int N_M = 1;   // 16 / MMA_M
    constexpr int N_N = 2;   // 16 / MMA_N
    constexpr int N_K = 4;   // 64 / MMA_K

    __shared__ half shared_q[16 * 64];
    __shared__ half shared_k[16 * 64];

    uint32_t reg_q[N_M][N_K][4];
    uint32_t reg_k[N_N][N_K][2];
    float reg_sim1[N_M][N_N][4] = {};
    
    // global to shared
    g2s<16, 64, BLOCK_SIZE>(shared_q, q + bid * l_q * d, d, tid);
    g2s<16, 64, BLOCK_SIZE>(shared_k, k + bid * l_kv * d, d, tid);
    asm volatile("cp.async.wait_all;");

    __syncthreads();

    // shared to register
    for (int i = 0; i < l_q / MMA_M; ++i) {
        for (int j = 0; j < d / MMA_K; ++j) {
            int row = lane_id % 16 + i * MMA_M;
            int col = lane_id / 16 * 8 + MMA_K * j;
            uint32_t addr = __cvta_generic_to_shared(&shared_q[row * d + swizzle(row, col)]);
            ldmatrix_x4(reg_q[i][j], addr);
        }
    }

    for (int i = 0; i < l_kv / MMA_N; ++i) {
        for (int j = 0; j < d / MMA_K; ++j) {
            int row = lane_id % 8 + i * MMA_N;
            int col = lane_id / 8 * 8 + j * MMA_K;
            uint32_t addr = __cvta_generic_to_shared(&shared_k[row * d + swizzle(row, col)]);
            ldmatrix_x2(reg_k[i][j], addr);
        }
    }

    __syncthreads();

    for (int i = 0; i < l_q / MMA_M; ++i) {
        for (int j = 0; j < l_kv / MMA_N; ++j) {
            for (int k = 0; k < d / MMA_K; ++k) {
                mma_m16n8k16(reg_q[i][k],
                             reg_k[j][k],
                             reg_sim1[i][j]);
            }
        }
    }

    for (int i = 0; i < l_q / MMA_M; ++i) {
        for (int j = 0; j < l_kv / MMA_N; ++j) {
            int offset = bid * l_q * l_kv;
            int row0 = lane_id >> 2;
            int row1 = row0 + 8;
            int col0 = lane_id % 4 * 2 + j * MMA_N;
            int col1 = col0 + 1;
            o[offset + row0 * l_kv + col0] = __float2half(reg_sim1[i][j][0] * scale);
            o[offset + row0 * l_kv + col1] = __float2half(reg_sim1[i][j][1] * scale);
            o[offset + row1 * l_kv + col0] = __float2half(reg_sim1[i][j][2] * scale);
            o[offset + row1 * l_kv + col1] = __float2half(reg_sim1[i][j][3] * scale);
        }
    }
}

torch::Tensor fusion(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor rel1, torch::Tensor rel2, torch::Tensor mask) {
    const int b = q.size(0);
    const int l_q = q.size(1);
    const int l_kv = k.size(1);
    const int d = q.size(2);

    auto o = torch::zeros({b, l_q, l_kv}, q.options());

    dim3 grid = {(unsigned int)b};
    dim3 block = {BLOCK_SIZE};

    fusion_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(o.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(rel1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(rel2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(mask.data_ptr<at::Half>()),
        b, l_q, l_kv, d);

    return o;
}

