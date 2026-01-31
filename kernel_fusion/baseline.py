import torch
from torch import Tensor
from torch.utils.benchmark import Timer

def fusion_kernel(q: Tensor, k: Tensor, v: Tensor, rel1: Tensor, rel2: Tensor, mask: Tensor) -> Tensor:
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

    s = torch.softmax(sim1 + sim2 + mask, dim=-1)

    out1 = torch.matmul(s, v)
    out2 = torch.einsum('b i j, j k d -> b i d', s, rel2)

    return out1 + out2

def fusion_kernel_cat(q: Tensor, k: Tensor, v: Tensor, rel1: Tensor, rel2: Tensor, mask: Tensor) -> Tensor:
    scale = q.size(-1) ** (-0.5)

    b = q.size(0)
    l_q = q.size(1)
    d = q.size(2)
    l_kv = k.size(1)

    tq = q.unsqueeze(-2)
    expand_q = tq.expand(b, l_q, l_kv, d)
    trel1 = rel1.unsqueeze(0)
    expand_rel1 = trel1.expand(b, l_q, l_kv, d)

if __name__ == "__main__":
    b, l_q, l_kv, d = 6400, 16, 16, 64

    q = torch.randn(b, l_q, d).to('cuda')
    k = torch.randn(b, l_kv, d).to('cuda')
    v = torch.randn(b, l_kv, d).to('cuda')
    rel1 = torch.randn(l_q, l_kv, d).to('cuda')
    rel2 = torch.randn(l_q, l_kv, d).to('cuda')

    p = 0.1
    mask = (torch.rand(b, l_q, l_kv, device=q.device) < p).to(q.dtype) * float('-inf')

    out = Timer(
        stmt='fusion_kernel(q, k, v, rel1, rel2, mask)',
        globals={'fusion_kernel': fusion_kernel, 'q': q, 'k': k, 'v': v, 'rel1': rel1, 'rel2': rel2, 'mask': mask},
    ).blocked_autorange(min_run_time=1.0)

    print(out)