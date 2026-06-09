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
import torch
import triton
import triton.language as tl

@triton.jit
def _dsd_matmul_helper(values_ptr, row_offsets_ptr, column_indices_ptr,
                      B_ptr, stride_bk, stride_bn, C_ptr, stride_cm, stride_cn,
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
    C_offs = m_offs*stride_cm + n_offs*stride_cn
    
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
            B_offs = outer_k_offs*stride_bk + n_offs*stride_bn
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
    grid = (M//BLOCK_M, triton.cdiv(N, BLOCK_N))

    _dsd_matmul_helper[grid](values, row_offsets, column_indices, B, B.stride(0),
                            B.stride(1), C, C.stride(0), C.stride(1), M, N, K,
                            BLOCK = block, BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N,
                            BLOCK_K = BLOCK_K)

    return C


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

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse_flash_forward (A2)")


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

    TODO: implement.
    """
    raise NotImplementedError("TODO: implement sparse_flash_backward (A3)")
