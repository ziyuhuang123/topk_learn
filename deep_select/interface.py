import functools
import torch

from typing import Any, Dict, Optional, Tuple

from . import deep_select_cuda as _backend


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    return _backend.get_alignment_requirement()


def get_cluster_capability(cluster_size: int) -> Dict[str, Any]:
    """Return launch attributes and occupancy for an exact cluster specialization."""
    if cluster_size not in (2, 4, 8):
        raise ValueError("cluster_size must be one of 2, 4, or 8")
    return _backend.get_cluster_capability(cluster_size)


def _prepare_topk_call(
    input: torch.Tensor,
    topk: int,
    sorted: bool,
    begin: Optional[torch.Tensor],
    end: Optional[torch.Tensor],
    indices_type: torch.dtype,
    sorted_index: bool,
    hint: Optional[torch.Tensor],
    output_idx: Optional[torch.Tensor],
    output_idx_offset: Optional[torch.Tensor],
    idx_oob_fill_value: int,
    value_oob_fill_value: float,
    return_value: bool,
    abort_when_nan_found: bool,
):
    if sorted:
        raise ValueError("DeepSelect only supports sorted=False in this BF16 build")
    if input.dtype != torch.bfloat16:
        raise TypeError(
            f"DeepSelect only supports torch.bfloat16 input; got {input.dtype}"
        )
    if not input.is_cuda:
        raise ValueError("DeepSelect input must be a CUDA tensor")
    if input.ndim != 2:
        raise ValueError(f"DeepSelect input must be 2D; got {input.ndim} dimensions")
    if input.stride(1) != 1:
        raise ValueError("DeepSelect input must be contiguous in its last dimension")
    if indices_type not in (torch.int32, torch.int64):
        raise TypeError("indices_type must be torch.int32 or torch.int64")

    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, device: torch.device, dtype: torch.dtype):
        """
        Return a tensor with shape (dim0, dim1), and stride (X, 1), where X is a multiple of 32B
        """
        output_stride_requirement_bytes = get_stride_requirement()[1]
        output_stride_requirement = output_stride_requirement_bytes // dtype.itemsize
        assert output_stride_requirement > 0
        dim1_rounded = (dim1+output_stride_requirement-1) // output_stride_requirement * output_stride_requirement
        return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]

    output_val = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=input.dtype) if return_value else None
    if output_idx is None:
        output_idx = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=indices_type)
    else:
        assert output_idx.dtype == indices_type

    assert begin is None, "`begin` is not supported now"
    assert hint is None, "`hint` is not supported now"
    backend_args = (
        input,
        topk,
        begin, end,
        sorted, sorted_index,
        output_val, output_idx,
        output_idx_offset,
        idx_oob_fill_value,
        value_oob_fill_value,
        return_value,
        abort_when_nan_found,
    )
    return output_val, output_idx, backend_args


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Select an unsorted Top-K from each row of a CUDA BF16 tensor.

    ``input`` must be a two-dimensional CUDA BF16 tensor whose last dimension
    is contiguous and whose row stride meets ``get_stride_requirement()``.
    This H20 teaching build only supports ``sorted=False``. ``indices_type``
    may be ``torch.int32`` or ``torch.int64``. Set ``return_value=False`` to
    omit values; the function still returns a
    ``(values_or_none, indices)`` tuple.

    ``begin`` and ``hint`` are not supported. ``end`` and
    ``output_idx_offset`` are optional contiguous CUDA int32 tensors with one
    element per input row. Output rows are internally padded to meet the CUDA
    kernel's alignment requirement, so returned tensors are not guaranteed to
    be contiguous for every ``topk``.
    """
    output_val, output_idx, backend_args = _prepare_topk_call(
        input, topk, sorted, begin, end, indices_type, sorted_index, hint,
        output_idx, output_idx_offset, idx_oob_fill_value,
        value_oob_fill_value, return_value, abort_when_nan_found,
    )
    _backend.topk(*backend_args)
    return output_val, output_idx


def benchmark_topk(
    input: torch.Tensor,
    topk: int,
    variant: str,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run a named DeepSelect V3 benchmark variant under the fair contract."""
    if topk > 512:
        raise ValueError("benchmark_topk requires topk <= 512")
    if indices_type != torch.int64:
        raise TypeError("benchmark_topk requires indices_type=torch.int64")
    if sorted_index:
        raise ValueError("benchmark_topk requires sorted_index=False")
    if not return_value:
        raise ValueError("benchmark_topk requires return_value=True")
    if variant in {"v3f_cluster_c2", "v3f_cluster_c4", "v3f_cluster_c8"} and abort_when_nan_found:
        raise ValueError("cluster benchmark variants require abort_when_nan_found=False")

    output_val, output_idx, backend_args = _prepare_topk_call(
        input, topk, sorted, begin, end, indices_type, sorted_index, hint,
        output_idx, output_idx_offset, idx_oob_fill_value,
        value_oob_fill_value, return_value, abort_when_nan_found,
    )
    _backend.topk_variant(*backend_args, variant)
    return output_val, output_idx
