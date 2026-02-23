import torch
from torch import Tensor
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load

fusion_kernel = load(
    name="fusion_kernel",
    sources=["pybind.cpp", "fusion.cu"],
    verbose=True
)

def fusion_baseline(q: Tensor, k: Tensor, v: Tensor, rel1: Tensor, rel2: Tensor, mask: Tensor) -> Tensor:
    '''
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
    '''
    scale = q.size(-1) ** (-0.5)

    sim1 = torch.einsum('b i d, b j d -> b i j', q, k) * scale
    sim2 = torch.einsum('b i d, i j d -> b i j', q, rel1) * scale

    s = torch.softmax(sim1 + sim2, dim=-1).to(q.dtype) + mask

    out1 = torch.matmul(s, v)
    out2 = torch.einsum('b i j, i j d -> b i d', s, rel2)

    return sim1 # out1 + out2

if __name__ == "__main__":
    b, l_q, l_kv, d = 6400, 16, 16, 64

    q = torch.randn(b, l_q, d, device='cuda')
    k = torch.randn(b, l_kv, d, device='cuda')
    v = torch.randn(b, l_kv, d, device='cuda')
    rel1 = torch.randn(l_q, l_kv, d, device='cuda')
    rel2 = torch.randn(l_q, l_kv, d, device='cuda')

    p = 0.1
    mask = torch.zeros(b, l_q, l_kv, device=q.device, dtype=q.dtype)
    mask.masked_fill_(torch.rand(b, l_q, l_kv, device=q.device) < p, float('-inf'))

    # 正确性检验
    out_baseline = fusion_baseline(q, k, v, rel1, rel2, mask)
    out_custom = fusion_kernel.fusion(q, k, v, rel1, rel2, mask)

    max_diff = (out_baseline - out_custom).abs().max().item()
    mean_diff = (out_baseline - out_custom).abs().mean().item()
    allclose = torch.allclose(out_baseline, out_custom, atol=1e-3, rtol=1e-3)

    print(f"=== 正确性检验 ===")
    print(f"Max  absolute diff: {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    print(f"torch.allclose(atol=1e-3, rtol=1e-3): {allclose}")
    print()

    # 性能对比
    globals_dict = {
        'fusion_baseline': fusion_baseline,
        'fusion_kernel': fusion_kernel,
        'q': q, 'k': k, 'v': v,
        'rel1': rel1, 'rel2': rel2, 'mask': mask,
    }

    t_baseline = Timer(
        stmt='fusion_baseline(q, k, v, rel1, rel2, mask)',
        globals=globals_dict,
    ).blocked_autorange(min_run_time=1.0)

    t_custom = Timer(
        stmt='fusion_kernel.fusion(q, k, v, rel1, rel2, mask)',
        globals=globals_dict,
    ).blocked_autorange(min_run_time=1.0)

    print(f"=== 性能对比 ===")
    print(f"Baseline (PyTorch): {t_baseline}")
    print(f"Custom   (CUDA):    {t_custom}")
    print(f"Speedup: {t_baseline.median / t_custom.median:.2f}x")