#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

#define BLOCK_SIZE 128

__global__ void fusion_kernel(const half* q, const half* k, 
                              const half* v, half* o,
                              const half* rel1, const half* rel2, 
                              const half* mask,
                              int b, int l_q, int l_kv, int d) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float scale = rsqrtf((float)d);

    __shared__ half shared_q[1024];
    __shared__ half shared_k[1024];
    __shared__ half shared_sim1[256];
    
    for (int i = tid; i < 1024; i += BLOCK_SIZE) {
        shared_q[i] = q[bid * l_q * d + i];
    }

    for (int i = tid; i < 1024; i += BLOCK_SIZE) {
        shared_k[i] = k[bid * l_kv * d + i];
    }

    __syncthreads();

    for (int i = tid; i < l_q * l_kv; i += BLOCK_SIZE) {
        int row = i / l_kv;
        int col = i % l_kv;
        float sum = 0.0f;
        for (int j = 0; j < d; ++j) {
            sum += __half2float(shared_q[row * d + j]) * __half2float(shared_k[col * d + j]);
        }
        shared_sim1[i] = __float2half(sum * scale);
    }

    for (int i = tid; i < 256; i += BLOCK_SIZE) {
        o[bid * l_q * l_kv + i] = shared_sim1[i];
    }
}

torch::Tensor fusion(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor rel1, torch::Tensor rel2, torch::Tensor mask) {
    int b = q.size(0);
    int l_q = q.size(1);
    int l_kv = k.size(1);
    int d = q.size(2);

    auto o = torch::zeros({b, l_q, l_kv}, q.options());

    dim3 grid = {(unsigned int)b};
    dim3 block = {128};

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

