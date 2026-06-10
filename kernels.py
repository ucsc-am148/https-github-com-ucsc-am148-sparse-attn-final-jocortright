"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dsd_matmul_helper(values_ptr, row_offsets_ptr, column_indices_ptr,
                      B_ptr, bk_stride, bn_stride, C_ptr, cm_stride, cn_stride,
                      M, N, K, BLOCK: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # program ids
    m_pid = tl.program_id(axis = 0)
    n_pid = tl.program_id(axis = 1)

    # calculate number of tiles and offsets within
    num_tiles = BLOCK//BLOCK_M
    tile_row = m_pid//num_tiles
    inner_row = m_pid % num_tiles

    # offsets
    m_offs = m_pid*BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_offs = n_pid*BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    C_offs = m_offs*cm_stride + n_offs*cn_stride
    
    # for inside the loop
    inner_m_offs = inner_row*BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    
    # start and end row offsets
    row_offsets_start_offs = tile_row
    row_offsets_end_offs = tile_row + 1
    
    # masks
    m_mask = m_offs < M
    n_mask = n_offs < N
    inner_m_mask = inner_m_offs < BLOCK
    C_mask = m_mask & n_mask
    
    # load row offset range
    row_offsets_start = tl.load(row_offsets_ptr + row_offsets_start_offs)
    row_offsets_end = tl.load(row_offsets_ptr + row_offsets_end_offs)
    
    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for column_indices_offs in range (row_offsets_start, row_offsets_end):
        # load column indices
        column_indices = tl.load(column_indices_ptr + column_indices_offs)

        for k in tl.static_range(BLOCK//BLOCK_K):
            # offsets for k
            k_offs = k*BLOCK_K + tl.arange(0, BLOCK_K)
            inner_k_offs = k_offs[None, :]
            outer_k_offs = column_indices*BLOCK + k_offs[:, None]
            
            # masks for k
            inner_k_mask = inner_k_offs < BLOCK
            outer_k_mask = outer_k_offs < K

            # define offsets and mask, then load values. this is our A matrix
            values_offs = column_indices_offs*BLOCK*BLOCK + inner_m_offs*BLOCK + inner_k_offs
            values_mask = inner_m_mask & inner_k_mask
            values = tl.load(values_ptr + values_offs, mask = values_mask, other = 0.0)

            # define offsets and mask, then load B matrix
            B_offs = outer_k_offs*bk_stride + n_offs*bn_stride
            B_mask = outer_k_mask & n_mask
            B = tl.load(B_ptr + B_offs, mask = B_mask, other = 0.0)

            acc += tl.dot(values, B, allow_tf32 = False)

    tl.store(C_ptr + C_offs, acc, mask = C_mask)

def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.
    """
    # define block sizes
    BLOCK_M = min(block, 64)
    BLOCK_N = min(block, 64)
    BLOCK_K = min(block, 32)
    
    # initialize output matrices
    C = torch.zeros((M, N), device = B.device, dtype = torch.float32)

    # grid size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # function call
    _dsd_matmul_helper[grid](values, row_offsets, column_indices, B, B.stride(0),
                            B.stride(1), C, C.stride(0), C.stride(1), M, N, K,
                            BLOCK = block, BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N,
                            BLOCK_K = BLOCK_K)

    return C


@triton.jit
def _sparse_flash_forward_helper(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, bh_stride,
                                t_stride, d_stride, l_stride, q_row_offsets_ptr,
                                q_col_indices_ptr, sm_scale, T, d,
                                BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                                BLOCK_D: tl.constexpr, LOG2E: tl.constexpr):
    # program ids
    bh_pid = tl.program_id(axis = 0)
    q_pid = tl.program_id(axis = 1)

    # offsets
    q_offs = q_pid*BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    d_trans_offs = tl.arange(0, BLOCK_D)[:, None]
    d_offs = tl.arange(0, BLOCK_D)[None, :]
    Q_offs = bh_pid*bh_stride + q_offs*t_stride + d_offs*d_stride

    # start and end q row offsets
    q_row_offsets_start_offs = q_pid
    q_row_offsets_end_offs = q_pid + 1
    
    # masks
    q_mask = q_offs < T
    d_mask = d_offs < d
    d_trans_mask = d_trans_offs < d
    Q_mask = q_mask & d_mask

    # load Q
    Q = tl.load(Q_ptr + Q_offs, mask = Q_mask, other = 0.0)

    # load q row offset range
    q_row_offsets_start = tl.load(q_row_offsets_ptr + q_row_offsets_start_offs)
    q_row_offsets_end = tl.load(q_row_offsets_ptr + q_row_offsets_end_offs)

    # calculate scale factor
    qkt_scale = sm_scale*LOG2E

    # initialize accumulators
    max_i = tl.full((BLOCK_Q,), float('-inf'), dtype = tl.float32)
    L_i = tl.zeros((BLOCK_Q,), dtype = tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype = tl.float32)

    for q_col_indices_offs in range (q_row_offsets_start, q_row_offsets_end):
        # load q column indices
        q_col_indices = tl.load(q_col_indices_ptr + q_col_indices_offs)
        
        # define offsets and masks for k
        k_offs = q_col_indices*BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
        k_trans_offs = q_col_indices*BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        k_mask = k_offs < T
        k_trans_mask = k_trans_offs < T
        
        # define V offset and mask
        V_offs = bh_pid*bh_stride + k_offs*t_stride + d_offs*d_stride
        V_mask = k_mask & d_mask

        # K needs to be loaded transposed
        KT_offs = bh_pid*bh_stride + d_trans_offs*d_stride + k_trans_offs*t_stride
        KT_mask = d_trans_mask & k_trans_mask

        # load K and V
        KT = tl.load(K_ptr + KT_offs, mask = KT_mask, other = 0.0)
        V = tl.load(V_ptr + V_offs, mask = V_mask, other = 0.0)

        # calculate scores
        S_i = tl.dot(Q, KT)*qkt_scale
        S_i = tl.where(k_trans_mask, S_i, float('-inf'))

        # softmax
        S_max = tl.max(S_i, axis = 1)
        max_new = tl.maximum(max_i, S_max)
        P = tl.exp2(S_i - max_new[:, None])
        P_sum = tl.sum(P, axis = 1)
        sqrt_d = tl.exp2(max_i - max_new)
        L_new = L_i*sqrt_d + P_sum

        # convert to fp16
        P = P.to(tl.float16)

        # update accumulators
        acc = sqrt_d[:, None]*acc + tl.dot(P, V)
        max_i = max_new
        L_i = L_new
        
    # calculate output matrix
    O = acc/(L_i[:, None])

    # calculate residuals
    L = tl.log2(L_i) + max_i

    # O output offset and mask
    O_offs = bh_pid*bh_stride + q_offs*t_stride + d_offs*d_stride
    O_mask = Q_mask

    # L residuals offset and mask
    l_offs = q_pid*BLOCK_Q + tl.arange(0, BLOCK_Q)
    l_mask = l_offs < T
    L_offs = bh_pid*l_stride + l_offs
    L_mask = l_mask

    # convert outputs to correct type
    O = O.to(tl.float16)
    L = L.to(tl.float32)
    
    # store
    tl.store(O_ptr + O_offs, O, mask = O_mask)
    tl.store(L_ptr + L_offs, L, mask = L_mask)

  
def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.
    """
    # access dimensions
    B, H, T, d = Q.shape

    # define constants
    BH = B*H
    LOG2E = math.log2(math.e)
    BLOCK_D = 2**math.ceil(math.log2(d))

    # flatten matrices
    Q_flat = Q.reshape(BH, T, d)
    K_flat = K.reshape(BH, T, d)
    V_flat = V.reshape(BH, T, d)

    # initialize output matrices
    O_flat = torch.zeros((BH, T, d), device = Q.device, dtype = torch.float32)
    L_flat = torch.zeros((BH, T), device = Q.device, dtype = torch.float32)

    # grid size
    grid = (BH, triton.cdiv(T, BLOCK_Q))

    # function call
    _sparse_flash_forward_helper[grid](Q_flat, K_flat, V_flat, O_flat, L_flat,
                                Q_flat.stride(0), Q_flat.stride(1),
                                Q_flat.stride(2), L_flat.stride(0),
                                q_row_offsets, q_col_indices,
                                sm_scale, T, d, BLOCK_Q = BLOCK_Q,
                                BLOCK_K = BLOCK_K, BLOCK_D = BLOCK_D,
                                LOG2E = LOG2E)
    
    # reshape output matrices
    O = O_flat.reshape(B, H, T, d)
    L = L_flat.reshape(B, H, T)

    return O, L


@triton.jit
def _sparse_flash_backward_d_helper(O_ptr, dO_ptr, D_ptr, bh_stride, t_stride,
                                   d_stride, D_stride, T, d, BLOCK_Q: tl.constexpr,
                                   BLOCK_D: tl.constexpr):
    # program ids
    bh_pid = tl.program_id(axis = 0)
    q_pid = tl.program_id(axis = 1)

    # offsets
    q_offs = q_pid*BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)[None, :]
    offs = bh_pid*bh_stride + q_offs[:, None]*t_stride + d_offs*d_stride
    D_offs = bh_pid*D_stride + q_offs

    # masks
    q_mask = q_offs < T
    d_mask = d_offs < d
    mask = q_mask[:, None] & d_mask

    # load matrices
    O = tl.load(O_ptr + offs, mask = mask, other = 0.0)
    dO = tl.load(dO_ptr + offs, mask = mask, other = 0.0)

    # calculate D_i
    D = tl.sum(O*dO, axis = 1)
    tl.store(D_ptr + D_offs, D, mask = q_mask)

@triton.jit
def _sparse_flash_backward_dq_helper(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
                                    dO_ptr, dQ_ptr, bh_stride, t_stride,
                                    d_stride, l_stride, q_row_offsets_ptr,
                                    q_col_indices_ptr, sm_scale, T, d,
                                    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                                    BLOCK_D: tl.constexpr, LOG2E: tl.constexpr):
    # program ids
    bh_pid = tl.program_id(axis = 0)
    q_pid = tl.program_id(axis = 1)

    # offsets
    q_offs = q_pid*BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
    l_offs = q_pid*BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_trans_offs = tl.arange(0, BLOCK_D)[:, None]
    d_offs = tl.arange(0, BLOCK_D)[None, :]
    Q_offs = bh_pid*bh_stride + q_offs*t_stride + d_offs*d_stride
    dO_offs = Q_offs
    L_offs = bh_pid*l_stride + l_offs
    D_offs = L_offs

    # start and end q row offsets
    q_row_offsets_start_offs = q_pid
    q_row_offsets_end_offs = q_pid + 1
    
    # masks
    q_mask = q_offs < T
    l_mask = l_offs < T
    d_mask = d_offs < d
    d_trans_mask = d_trans_offs < d
    Q_mask = q_mask & d_mask
    dO_mask = Q_mask
    L_mask = l_mask
    D_mask = L_mask

    # load matrices
    Q = tl.load(Q_ptr + Q_offs, mask = Q_mask, other = 0.0)
    dO = tl.load(dO_ptr + dO_offs, mask = dO_mask, other = 0.0)
    L = tl.load(L_ptr + L_offs, mask = L_mask, other = 0.0)
    D = tl.load(D_ptr + D_offs, mask = D_mask, other = 0.0)

    # load q row offset range
    q_row_offsets_start = tl.load(q_row_offsets_ptr + q_row_offsets_start_offs)
    q_row_offsets_end = tl.load(q_row_offsets_ptr + q_row_offsets_end_offs)

    # calculate scale factor
    qkt_scale = sm_scale*LOG2E

    # initialize accumulator
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype = tl.float32)

    for q_col_indices_offs in range (q_row_offsets_start, q_row_offsets_end):
        # load q column indices
        q_col_indices = tl.load(q_col_indices_ptr + q_col_indices_offs)

        # define offsets and masks for k
        k_offs = q_col_indices*BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
        k_trans_offs = q_col_indices*BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        k_mask = k_offs < T
        k_trans_mask = k_trans_offs < T
        K_offs = bh_pid*bh_stride + k_offs*t_stride + d_offs*d_stride
        K_mask = k_mask & d_mask
        
        # now V needs to be loaded transposed
        VT_offs = bh_pid*bh_stride + d_trans_offs*d_stride + k_trans_offs*t_stride
        VT_mask = d_trans_mask & k_trans_mask

        # load K and V, define KT
        K = tl.load(K_ptr + K_offs, mask = K_mask, other = 0.0)
        VT = tl.load(V_ptr + VT_offs, mask = VT_mask, other = 0.0)
        KT = tl.trans(K)

        # calculate scores
        S_i = tl.dot(Q, KT)*qkt_scale
        S_i = tl.where(k_trans_mask, S_i, float('-inf'))

        # recalculate P
        P = tl.exp2(S_i - L[:, None])
        
        # calculate dP and dS, convert dS to fp16
        dP = tl.dot(dO, VT)
        dS = P*(dP - D[:, None])
        dS = dS.to(tl.float16)

        # update accumulator
        acc += tl.dot(dS, K)*sm_scale
    
    # convert output to fp16
    acc = acc.to(tl.float16)
    
    tl.store(dQ_ptr + Q_offs, acc, mask = Q_mask)

@triton.jit
def _sparse_flash_backward_dkdv_helper(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
                                      dO_ptr, dK_ptr, dV_ptr, bh_stride,
                                      t_stride, d_stride, l_stride,
                                      k_row_offsets_ptr, k_col_indices_ptr,
                                      sm_scale, T, d, BLOCK_Q: tl.constexpr,
                                      BLOCK_K: tl.constexpr,
                                      BLOCK_D: tl.constexpr,
                                      LOG2E: tl.constexpr):
    # program ids
    bh_pid = tl.program_id(axis = 0)
    k_pid = tl.program_id(axis = 1)

    # offsets
    k_trans_offs = k_pid*BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    k_offs = k_pid*BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    d_trans_offs = tl.arange(0, BLOCK_D)[:, None]
    d_offs = tl.arange(0, BLOCK_D)[None, :]
    KT_offs = bh_pid*bh_stride + d_trans_offs*d_stride + k_trans_offs*t_stride
    VT_offs = KT_offs
    K_offs = bh_pid*bh_stride + k_offs*t_stride + d_offs*d_stride
    V_offs = K_offs
    
    # start and end q row offsets
    k_row_offsets_start_offs = k_pid
    k_row_offsets_end_offs = k_pid + 1
    
    # masks
    k_trans_mask = k_trans_offs < T
    k_mask = k_offs < T
    d_trans_mask = d_trans_offs < d
    d_mask = d_offs < d
    KT_mask = k_mask & d_mask
    VT_mask = KT_mask
    K_mask = k_mask & d_mask
    V_mask = K_mask

    # load matrices
    KT = tl.load(K_ptr + KT_offs, mask = KT_mask, other = 0.0)
    VT = tl.load(V_ptr + VT_offs, mask = VT_mask, other = 0.0)
    
    # load q row offset range
    k_row_offsets_start = tl.load(k_row_offsets_ptr + k_row_offsets_start_offs)
    k_row_offsets_end = tl.load(k_row_offsets_ptr + k_row_offsets_end_offs)

    # calculate scale factor
    qkt_scale = sm_scale*LOG2E

    # initialize accumulators
    dk_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype = tl.float32)
    dv_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype = tl.float32)

    for k_col_indices_offs in range (k_row_offsets_start, k_row_offsets_end):
        # load k column indices
        k_col_indices = tl.load(k_col_indices_ptr + k_col_indices_offs)

        # define offsets and masks for q
        q_offs = k_col_indices*BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]
        l_offs = k_col_indices*BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < T
        l_mask = l_offs < T
        
        # define offsets and masks for Q, dO, L, and D
        Q_offs = bh_pid*bh_stride + q_offs*t_stride + d_offs*d_stride
        Q_mask = q_mask & d_mask
        dO_offs = Q_offs
        dO_mask = Q_mask
        L_offs = bh_pid*l_stride + l_offs
        L_mask = l_mask
        D_offs = L_offs
        D_mask = L_mask

        # load matrices
        Q = tl.load(Q_ptr + Q_offs, mask = Q_mask, other = 0.0)
        dO = tl.load(dO_ptr + dO_offs, mask = dO_mask, other = 0.0)
        L = tl.load(L_ptr + L_offs, mask = L_mask, other = 0.0)
        D = tl.load(D_ptr + D_offs, mask = D_mask, other = 0.0)

        # calculate scores
        S_i = tl.dot(Q, KT)*qkt_scale
        S_i = tl.where(k_trans_mask, S_i, float('-inf'))

        # recalculate P, transpose, and convert to fp16
        P = tl.exp2(S_i - L[:, None])
        PT = tl.trans(P)
        PT = PT.to(tl.float16)

        # update dv accumulator
        dv_acc += tl.dot(PT, dO)
        
        # calculate dP and dS, transpose dS and convert to fp16
        dP = tl.dot(dO, VT)
        dS = P*(dP - D[:, None])
        dST = tl.trans(dS)
        dST = dST.to(tl.float16)

        # update dk accumulator
        dk_acc += tl.dot(dST, Q)*sm_scale
    
    # convert outputs to correct type
    dk_acc = dk_acc.to(tl.float16)
    dv_acc = dv_acc.to(tl.float16)

    tl.store(dK_ptr + K_offs, dk_acc, mask = K_mask)
    tl.store(dV_ptr + V_offs, dv_acc, mask = V_mask)

def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.
    """
    # access dimensions
    B, H, T, d = Q.shape

    # define constants
    BH = B*H
    LOG2E = math.log2(math.e)
    BLOCK_D = 2**math.ceil(math.log2(d))

    # flatten matrices
    Q_flat = Q.reshape(BH, T, d)
    K_flat = K.reshape(BH, T, d)
    V_flat = V.reshape(BH, T, d)
    O_flat = O.reshape(BH, T, d)
    dO_flat = dO.reshape(BH, T, d)
    L_flat = L.reshape(BH, T)

    # initialize supporting matrices
    D = torch.zeros((BH, T), device = Q.device, dtype = torch.float32)

    # initialize output matrices
    dQ_flat = torch.zeros((BH, T, d), device = Q.device, dtype = torch.float32)
    dK_flat = torch.zeros((BH, T, d), device = K.device, dtype = torch.float32)
    dV_flat = torch.zeros((BH, T, d), device = V.device, dtype = torch.float32)

    # grid size
    grid = (BH, triton.cdiv(T, BLOCK_Q))

    # function call to populate D
    _sparse_flash_backward_d_helper[grid](O_flat, dO_flat, D, Q_flat.stride(0),
                                         Q_flat.stride(1), Q_flat.stride(2),
                                         D.stride(0), T, d, BLOCK_Q = BLOCK_Q,
                                         BLOCK_D = BLOCK_D)

    # function call to populate dQ
    _sparse_flash_backward_dq_helper[grid](Q_flat, K_flat, V_flat, O_flat,
                                          L_flat, D, dO_flat, dQ_flat,
                                          Q_flat.stride(0), Q_flat.stride(1),
                                          Q_flat.stride(2), L_flat.stride(0),
                                          q_row_offsets, q_col_indices,
                                          sm_scale, T, d, BLOCK_Q = BLOCK_Q,
                                          BLOCK_K = BLOCK_K, BLOCK_D = BLOCK_D,
                                          LOG2E = LOG2E)
    
    # function call to populate dK and dV
    _sparse_flash_backward_dkdv_helper[grid](Q_flat, K_flat, V_flat, O_flat,
                                            L_flat, D, dO_flat, dK_flat,
                                            dV_flat, Q_flat.stride(0),
                                            Q_flat.stride(1), Q_flat.stride(2),
                                            L_flat.stride(0), k_row_offsets,
                                            k_col_indices, sm_scale, T, d,
                                            BLOCK_Q = BLOCK_Q, BLOCK_K = BLOCK_K,
                                            BLOCK_D = BLOCK_D, LOG2E = LOG2E)

    # reshape output matrices
    dQ = dQ_flat.reshape(B, H, T, d)
    dK = dK_flat.reshape(B, H, T, d)
    dV = dV_flat.reshape(B, H, T, d)

    return dQ, dK, dV
