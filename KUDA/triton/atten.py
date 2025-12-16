import triton
import triton.language as tl


@triton.jit
def attn_fwd(
    q_ptr,          # [M, D]
    k_ptr,          # [N, D]
    v_ptr,          # unused for now
    o_ptr,          # unused for now

    M,
    N,
    HEAD_DIM,

    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,

    sm_scale,
    causal: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    Q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm
              + tl.arange(0, HEAD_DIM)[None, :] * stride_qd,
        mask=offs_m[:, None] < M,
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        K = tl.load(
            k_ptr + offs_n[:, None] * stride_km
                  + tl.arange(0, HEAD_DIM)[None, :] * stride_kd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )

        qk = tl.dot(Q, tl.trans(K))


        qk *= sm_scale

        if causal:
            mask = offs_m[:, None] < offs_n[None, :]
            qk = tl.where(mask, -float("inf"), qk)

        block_max = tl.max(qk, axis=1)
        new_max = tl.maximum(m_i, block_max)

        alpha = tl.exp(m_i - new_max)
        l_i *= alpha

        p = tl.exp(qk - new_max[:, None])
        l_i += tl.sum(p, axis=1)

        m_i = new_max

        acc_o = tl.zeros((BLOCK_M, HEAD_DIM), float32)

        acc_o += p @ V_block

    output = acc_o / l_i[:, None]


