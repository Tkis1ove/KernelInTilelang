#include <cstdint>
#include <cuda_fp16.h>

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]),
                   "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void g2s(half* dst, const half* src, int src_stride, int tid) {
    constexpr int num_elems = 16 / sizeof(half);
    constexpr int num_iters = (HEIGHT * WIDTH) / (num_elems * TB_SIZE);

    for (int i = 0; i < num_iters; i++) {
        const int idx = (i * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        const uint32_t dst_addr = __cvta_generic_to_shared(&dst[idx]);
        const half* src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     : : "r"(dst_addr), "l"(src_addr));
    }
    asm volatile("cp.async.commit_group;");
}