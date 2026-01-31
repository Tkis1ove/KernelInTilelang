import tilelang
import tilelang.language as T
from tilelang import jit
import torch
import math

'''
q           :[b, l_q, d]
k, v        :[b, l_kv, d]
rel1, rel2  :[l_q, l_kv, d]
mask        :[b, l_q, l_kv]

sim1 = q @ k / sqrt(d)                                     :[b, l_q, l_kv]
sim2 = einsum("b i d, i j d -> b i j", q, rel1) / sqrt(d)  :[b, l_q, l_kv]
s = softmax(sim1 + sim2) + mask                            :[b, l_q, l_kv]

out1 = s @ v     :[b, l_q, d]
out2 = s @ rel2  :[b, l_q, d]

o = out1 + out2  :[b, l_q, d]

data future:
q.size() = k.size() = v.size()
l_q = l_kv = 16
d = 64
b = [800, 3200, 6400, 12800, 20480]
'''
@jit
def fusion(b: int, l_q: int, l_kv: int, d: int, dtype: str = "float16"):
    scale = 1.0 / math.sqrt(d)
    q_shape = [b, l_q, d]
    kv_shape = [b, l_kv, d]
    rel12_shape = [l_q, l_kv, d]
    mask_shape = [b, l_q, l_kv]
    accum_dtype = T.float32

    @T.prim_func
    def fusion_kernel(
        q: T.Tensor[q_shape, dtype],
        k: T.Tensor[kv_shape, dtype],
        v: T.Tensor[kv_shape, dtype],
        rel1: T.Tensor[rel12_shape, dtype],
        rel2: T.Tensor[rel12_shape, dtype],
        o: T.Tensor[q_shape, dtype],
        mask: T.Tensor[mask_shape, dtype]
    ):
        with T.Kernel(b, threads=64) as bx:
            q_shared = T.alloc_shared([l_q, d], dtype)
            k_shared = T.alloc_shared([l_kv, d], dtype)
            v_shared = T.alloc_shared([l_kv, d], dtype)
            rel1_shared = T.alloc_shared(rel12_shape, dtype)
            rel2_shared = T.alloc_shared(rel12_shape, dtype)
            sim1_shared = T.alloc_fragment([l_q, l_kv], dtype)
            sim2_shared = T.alloc_shared([l_q, l_kv], dtype)
            s_shared = T.alloc_shared([l_q, l_kv], accum_dtype)
            s_o = T.alloc_fragment([l_q, l_kv], accum_dtype)
            s_cast = T.alloc_shared([l_q, l_kv], dtype)
            scores_max = T.alloc_fragment([l_q], accum_dtype)
            logsum = T.alloc_fragment([l_q], accum_dtype)
            o_local = T.alloc_fragment([l_q, d], dtype)

            T.copy(q[bx, : , : ], q_shared)
            T.copy(k[bx, : , : ], k_shared)
            T.copy(rel1, rel1_shared)
            T.copy(rel2, rel2_shared)

            T.fill(sim2_shared, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # sim1 = q @ k / sqrt(d)
            T.gemm(q_shared, k_shared, sim1_shared, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(l_q, l_kv):
                sim1_shared[i, j] *= scale
            # sim2 = einsum("b i d, i j d -> b i j", q, rel1) / sqrt(d) 
            for i in T.serial(l_q):
                for j in T.serial(l_kv):
                    for k in T.Parallel(d):
                        sim2_shared[i, j] += q_shared[i, k] * rel1_shared[i, j, k]
                    sim2_shared[i, j] *= scale
            # s = softmax(sim1 + sim2) + mask
            for i, j in T.Parallel(l_q, l_kv):
                s_shared[i, j] = sim1_shared[i, j] + sim2_shared[i, j]
            # find max
            T.reduce_max(s_shared, scores_max, dim=-1, clear=False)
            # logsum
            for i, j in T.Parallel(l_q, l_kv):
                s_shared[i, j] = T.exp(s_shared[i, j] - scores_max[i])
            T.reduce_sum(s_shared, logsum, dim=-1)
            # rel
            for i, j in T.Parallel(l_q, l_kv):
                s_o[i, j] = s_shared[i, j] / logsum[i] + mask[bx, i, j]
            T.copy(s_o, s_cast)

            # out1 = s @ v
            T.copy(v[bx, : , : ], v_shared)
            T.gemm(s_cast, v_shared, o_local, policy=T.GemmWarpPolicy.FullRow)

            # out2 = einsum("b i j, i j d -> b i d", s, rel2)
            for i in T.serial(l_q):
                for dim in T.serial(d):
                    for j in T.Parallel(l_kv):
                        o_local[i, dim] += s_cast[i, j] * rel2_shared[i, j, dim]

            T.copy(o_local, o[bx, : , : ])

    return fusion_kernel

if __name__ == "__main__":
    b, l_q, l_kv, d = 6400, 16, 16, 64

    # 修复：全部用 .half() 确保 float16
    q = torch.randn(b, l_q, d, device='cuda', dtype=torch.float16)
    k = torch.randn(b, l_kv, d, device='cuda', dtype=torch.float16)
    v = torch.randn(b, l_kv, d, device='cuda', dtype=torch.float16)
    o = torch.zeros(b, l_q, d, device='cuda', dtype=torch.float16)
    rel1 = torch.randn(l_q, l_kv, d, device='cuda', dtype=torch.float16)
    rel2 = torch.randn(l_q, l_kv, d, device='cuda', dtype=torch.float16)

    p = 0.1
    # mask 也要用 float16
    mask = (torch.rand(b, l_q, l_kv, device='cuda', dtype=torch.float16) < p) * float('-inf')
    mask = mask.half()  # 确保是 half 类型

    fusion_kernel = fusion(b, l_q, l_kv, d)
    fusion_kernel(q, k, v, rel1, rel2, o, mask)
    
    print("Kernel executed successfully!")
    print(f"Output shape: {o.shape}")
    print(f"Output dtype: {o.dtype}")



