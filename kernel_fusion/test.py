import torch

b, l_q, l_kv, d = 2, 3, 4, 5

q = torch.randn(b, l_q, d)
rel1 = torch.randn(l_q, l_kv, d)

# 方法1: einsum (推荐，自动处理广播)
sim2_einsum = torch.einsum('bid,ijd->bij', q, rel1)
print(sim2_einsum.shape)  # [2, 3, 4]

# 方法2: 手动广播，验证你的理解
q_expanded = q.unsqueeze(2).expand(b, l_q, l_kv, d)      # [b, l_q, l_kv, d]
rel1_expanded = rel1.unsqueeze(0).expand(b, l_q, l_kv, d)  # [b, l_q, l_kv, d]

# 逐元素乘，然后在最后一维求和
sim2_manual = (q_expanded * rel1_expanded).sum(dim=-1)   # [b, l_q, l_kv]

print(torch.allclose(sim2_einsum, sim2_manual))  # True！