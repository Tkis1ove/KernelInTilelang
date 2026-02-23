#include <cuda_runtime.h>
#include <torch/extension.h>

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
*/

#define WARP_SIZE 32

__global__ void fusion_kernel(const float* q, const float* k, 
                              const float* v, float* o,
                              const float* rel1, const float* rel2, 
                              const float* mask,
                              int b, int l_q, int l_kv, int d) {
    int bid = blockIdx.x * 4 + threadIdx.y; // 负责处理的batch_id，每个block处理4个batch
    int tid = threadIdx.x;

    float scale = rsqrtf((float)d);

    float local_q[16][64];
    float local_k[16][64];
    
    for (int i = 0; i < l_q; ++i) {
        for (int j = 0; j < d; j += WARP_SIZE) {
            local_q[i][tid + j] = q[bid * l_q * d + i * d + tid + j];
        }
    }

    for (int i = 0; i < l_kv; ++i) {
        for (int j = 0; j < d; j += WARP_SIZE) {
            local_k[i][tid + j] = k[bid * l_kv * d + i * d + tid + j];
        }
    }

    __syncthreads();

    float sim1[16][16];
    for (int i = 0; i < l_q; ++i) {
        for (int j = 0; j < l_kv; ++j) {
            float sum = 0.0;
            for (int k = 0; k < d; k += WARP_SIZE) {
                sum += local_q[i][k + tid] * local_k[j][k + tid];
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (tid == 0)
                sim1[i][j] = sum * scale;
        }
    }

    for (int i = 0; i < l_q; ++i) {
        for (int j = tid; j < l_kv; j += WARP_SIZE) {
            o[bid * l_q * l_kv + i * l_kv + j] = sim1[i][j];
        }
    }
}

torch::Tensor fusion(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor rel1, torch::Tensor rel2, torch::Tensor mask) {
    int b = q.size(0);
    int l_q = q.size(1);
    int l_kv = k.size(1);
    int d = q.size(2);

    auto o = torch::zeros({b, l_q, l_kv}, q.options());

    dim3 grid = {(unsigned int)(b / 4)};
    dim3 block = {32, 4};

    fusion_kernel<<<grid, block>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), 
                                   o.data_ptr<float>(), rel1.data_ptr<float>(), rel2.data_ptr<float>(), mask.data_ptr<float>(),
                                   b, l_q, l_kv, d);

    return o;
}

