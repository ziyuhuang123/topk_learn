// This file contains only the host-side topk() function and pybind11 module.
// Kernel template instantiations are in separate files for parallel compilation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <string>

#include "kerutils/supplemental/torch_tensors.h"

#include "dispatch_utils.h"

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.h"
#include "cuda_kernels/v3_cluster/topk_select.h"

struct TopkCall {
    TopkSelectArgs args;
    at::ScalarType output_index_t;
    uint32_t num_waves;
};

using V3FClusterC2Config = TopkSelectConfig<
    nv_bfloat16, int64_t, false, false, true,
    512, 256, 1, 4096, 4096, 2, 512, 2>;
using V3FClusterC4Config = TopkSelectConfig<
    nv_bfloat16, int64_t, false, false, true,
    512, 256, 1, 4096, 4096, 4, 512, 4>;
using V3FClusterC8Config = TopkSelectConfig<
    nv_bfloat16, int64_t, false, false, true,
    512, 256, 1, 4096, 4096, 8, 512, 8>;

TopkCall prepare_topk_call(
    torch::Tensor &input,
    int topk,
    c10::optional<torch::Tensor> &begin,
    c10::optional<torch::Tensor> &end,
    bool sorted_value,
    bool sorted_index,
    c10::optional<torch::Tensor> &output_value,
    torch::Tensor &output_index,
    c10::optional<torch::Tensor> &output_idx_offset,
    int idx_oob_fill_value,
    float value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    int batch_size = input.size(0);
    int vocab_size = input.size(1);
    at::ScalarType value_t = input.scalar_type();
    at::ScalarType output_index_t = output_index.scalar_type();

    TORCH_CHECK(topk > 0, "topk must > 0");
    TORCH_CHECK(!sorted_value, "`sorted=True` is not supported; use `sorted=False`");
    TORCH_CHECK(value_t == at::kBFloat16, "input dtype must be torch.bfloat16");
    TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
    if (return_value) {
        TORCH_CHECK(output_value.has_value(), "`output_value` must not be `None` when `return_value` is True");
    }

    KU_CHECK_DEVICE(input);
    KU_CHECK_DEVICE(begin);
    KU_CHECK_DEVICE(end);
    KU_CHECK_DEVICE(output_value);
    KU_CHECK_DEVICE(output_index);
    KU_CHECK_DEVICE(output_idx_offset);

    KU_CHECK_SHAPE(input, batch_size, vocab_size);
    KU_CHECK_SHAPE(begin, batch_size);
    KU_CHECK_SHAPE(end, batch_size);
    KU_CHECK_SHAPE(output_value, batch_size, topk);
    KU_CHECK_SHAPE(output_index, batch_size, topk);
    KU_CHECK_SHAPE(output_idx_offset, batch_size);
    
    KU_CHECK_DTYPE(input, value_t);
    KU_CHECK_DTYPE(begin, at::kInt);
    KU_CHECK_DTYPE(end, at::kInt);
    KU_CHECK_DTYPE(output_value, value_t);
    KU_CHECK_DTYPE(output_index, output_index_t);
    KU_CHECK_DTYPE(output_idx_offset, at::kInt);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(input);
    KU_CHECK_CONTIGUOUS(begin);
    KU_CHECK_CONTIGUOUS(end);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_value);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_index);
    KU_CHECK_CONTIGUOUS(output_idx_offset);

    auto check_dim0_stride = [&](const char tensor_name[], torch::Tensor &tensor, uint32_t alignment_requirement_bytes) {
        int64_t cur_stride = tensor.stride(0);
        uint64_t itemsize = tensor.dtype().itemsize();
        TORCH_CHECK(
            cur_stride * itemsize % alignment_requirement_bytes == 0,
            tensor_name, ".stride(0) (currently ", cur_stride,
            " numbers) must be a multiple of ", alignment_requirement_bytes,
            " Bytes (", alignment_requirement_bytes / itemsize, " numbers)"
        );
    };
    check_dim0_stride("input", input, INPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    check_dim0_stride("output_index", output_index, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    if (output_value.has_value()) {
        check_dim0_stride("value", *output_value, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    }

    constexpr uint64_t TMA_BASE_ALIGNMENT_BYTES = 16;
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(input.data_ptr()) % TMA_BASE_ALIGNMENT_BYTES == 0,
        "input data pointer must be aligned to ", TMA_BASE_ALIGNMENT_BYTES, " Bytes"
    );
    TORCH_CHECK(input.stride(0) >= vocab_size, "input rows must not overlap");
    const uint64_t input_storage_elements =
        input.storage().nbytes() / input.dtype().itemsize();
    const uint64_t required_input_elements =
        static_cast<uint64_t>(input.storage_offset()) +
        static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(input.stride(0));
    TORCH_CHECK(
        required_input_elements <= input_storage_elements,
        "input storage must cover every complete aligned row because TMA may read "
        "past the logical row width"
    );

    cudaDeviceProp* device_prop = at::cuda::getDeviceProperties(at::cuda::current_device());
    TORCH_CHECK(device_prop != nullptr);
    TopkSelectArgs args = {
        (uint32_t)batch_size,
        (uint32_t)vocab_size,
        (uint32_t)topk,

        input.data_ptr(),
        ku::get_optional_tensor_ptr<void>(output_value),
        output_index.data_ptr(),
        ku::get_optional_tensor_ptr<int>(begin),
        ku::get_optional_tensor_ptr<int>(end),
        ku::get_optional_tensor_ptr<int>(output_idx_offset),

        (uint64_t)input.stride(0),
        output_value.has_value() ? (uint64_t)output_value->stride(0) : 0,
        (uint64_t)output_index.stride(0),

        sorted_value,
        sorted_index,
        return_value,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,

        device_prop->sharedMemPerBlockOptin,
        at::cuda::getCurrentCUDAStream().stream()
    };

    uint32_t num_sm = device_prop->multiProcessorCount;
    uint32_t num_waves = (batch_size + num_sm-1) / num_sm;

    TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                "vocab_size must be < 2^23 for bfloat16 input");
    TORCH_CHECK(topk <= 4096, "topk must be <= 4096");

    return {args, output_index_t, num_waves};
}

void dispatch_topk(const TopkCall &call) {
    const TopkSelectArgs &args = call.args;
    uint32_t topk = args.topk;

    INTEGER_TYPE_SWITCH(call.output_index_t, OutIdxT, [&]() {
        BOOL_SWITCH(args.sorted_index, SORTED_INDEX, [&]() {
            BOOL_SWITCH(args.return_value, RETURN_VALUE, [&]() {
                //   wave == 1 -> occ1 (512t / B8192 / B2 4096 / TMA5 rounds)
                //   otherwise -> occ2 (256t / B4096 / B2 4096 / TMA3 or 4)
                auto dispatch = [&]<uint32_t MAX_TOPK>() {
                    if (call.num_waves == 1)
                        topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 512, 1, 8192, 4096, 5>>(args);
                    else if constexpr (MAX_TOPK <= 512)
                        topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 4>>(args);
                    else
                        topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 2, 4096, 4096, 3>>(args);
                };
                if (topk <= 512) {
                    dispatch.template operator()<512>();
                } else if (topk <= 1024) {
                    dispatch.template operator()<1024>();
                } else {
                    // Big-topk coverage tier, topk in (1024, 4096]: one correctness-only tuple
                    // (512t / occ1 / B8192 / B2 4096 / TMA3 / max_topk 4096), no wave split.
                    topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, 4096, 512, 1, 8192, 4096, 3>>(args);
                }
            });
        });
    });
}

void topk(
    torch::Tensor &input,
    int topk,
    c10::optional<torch::Tensor> &begin,
    c10::optional<torch::Tensor> &end,
    bool sorted_value,
    bool sorted_index,
    c10::optional<torch::Tensor> &output_value,
    torch::Tensor &output_index,
    c10::optional<torch::Tensor> &output_idx_offset,
    int idx_oob_fill_value,
    float value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    dispatch_topk(prepare_topk_call(
        input, topk, begin, end, sorted_value, sorted_index,
        output_value, output_index, output_idx_offset,
        idx_oob_fill_value, value_oob_fill_value,
        return_value, abort_when_nan_found));
}

void topk_variant(
    torch::Tensor &input,
    int topk,
    c10::optional<torch::Tensor> &begin,
    c10::optional<torch::Tensor> &end,
    bool sorted_value,
    bool sorted_index,
    c10::optional<torch::Tensor> &output_value,
    torch::Tensor &output_index,
    c10::optional<torch::Tensor> &output_idx_offset,
    int idx_oob_fill_value,
    float value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found,
    const std::string &variant
) {
    TopkCall call = prepare_topk_call(
        input, topk, begin, end, sorted_value, sorted_index,
        output_value, output_index, output_idx_offset,
        idx_oob_fill_value, value_oob_fill_value,
        return_value, abort_when_nan_found);

    if (variant == "v3e_deepselect") {
        dispatch_topk(call);
        return;
    }

    const bool is_cluster_variant =
        variant == "v3f_cluster_c2" ||
        variant == "v3f_cluster_c4" ||
        variant == "v3f_cluster_c8";
    const bool known_variant =
        variant == "v3a_scan_filter_atomic" ||
        variant == "v3b_ballot_compaction" ||
        variant == "v3c_tma_pipeline" ||
        variant == "v3d_adaptive_threshold" ||
        is_cluster_variant;
    TORCH_CHECK(known_variant, "Unknown topk variant: ", variant);
    TORCH_CHECK(call.args.topk <= 512, "Benchmark topk variants require topk <= 512");
    TORCH_CHECK(call.output_index_t == at::kLong, "Benchmark topk variants require torch.int64 output indices");
    TORCH_CHECK(!call.args.sorted_index, "Benchmark topk variants require sorted_index=False");
    TORCH_CHECK(call.args.return_value, "Benchmark topk variants require return_value=True");

    if (is_cluster_variant) {
        TORCH_CHECK(!call.args.abort_when_nan_found,
                    "Cluster benchmark variants require abort_when_nan_found=False");
        if (variant == "v3f_cluster_c2") {
            topk_select_bf16_cluster::run_topk_select_kernel<V3FClusterC2Config>(call.args);
        } else if (variant == "v3f_cluster_c4") {
            topk_select_bf16_cluster::run_topk_select_kernel<V3FClusterC4Config>(call.args);
        } else {
            topk_select_bf16_cluster::run_topk_select_kernel<V3FClusterC8Config>(call.args);
        }
        return;
    }

    using V3AConfig = TopkSelectConfig<
        nv_bfloat16, int64_t, false, false, true,
        512, 256, 2, 4096, 4096, 2, 512, 1,
        CandidateCompactionPolicy::SharedAtomic,
        TmaSchedulePolicy::Serial,
        ReconstructPolicy::Fixed>;
    using V3BConfig = TopkSelectConfig<
        nv_bfloat16, int64_t, false, false, true,
        512, 256, 2, 4096, 4096, 2, 512, 1,
        CandidateCompactionPolicy::Ballot,
        TmaSchedulePolicy::Serial,
        ReconstructPolicy::Fixed>;
    using V3CConfig = TopkSelectConfig<
        nv_bfloat16, int64_t, false, false, true,
        512, 256, 2, 4096, 4096, 4, 512, 1,
        CandidateCompactionPolicy::Ballot,
        TmaSchedulePolicy::Pipelined,
        ReconstructPolicy::Fixed>;
    using V3DConfig = TopkSelectConfig<
        nv_bfloat16, int64_t, false, false, true,
        512, 256, 2, 4096, 4096, 4, 512, 1,
        CandidateCompactionPolicy::Ballot,
        TmaSchedulePolicy::Pipelined,
        ReconstructPolicy::Adaptive>;

    if (variant == "v3a_scan_filter_atomic") {
        topk_select_bf16_normal::run_topk_select_kernel<V3AConfig>(call.args);
    } else if (variant == "v3b_ballot_compaction") {
        topk_select_bf16_normal::run_topk_select_kernel<V3BConfig>(call.args);
    } else if (variant == "v3c_tma_pipeline") {
        topk_select_bf16_normal::run_topk_select_kernel<V3CConfig>(call.args);
    } else {
        topk_select_bf16_normal::run_topk_select_kernel<V3DConfig>(call.args);
    }
}

std::pair<uint32_t, uint32_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

pybind11::dict get_cluster_capability(int cluster_size) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    topk_select_bf16_cluster::ClusterCapability capability;
    if (cluster_size == 2) {
        capability = topk_select_bf16_cluster::get_cluster_capability<V3FClusterC2Config>(stream);
    } else if (cluster_size == 4) {
        capability = topk_select_bf16_cluster::get_cluster_capability<V3FClusterC4Config>(stream);
    } else if (cluster_size == 8) {
        capability = topk_select_bf16_cluster::get_cluster_capability<V3FClusterC8Config>(stream);
    } else {
        TORCH_CHECK(false, "cluster_size must be one of 2, 4, or 8; got ", cluster_size);
    }

    pybind11::dict result;
    result["requested_cluster_size"] = capability.requested_cluster_size;
    result["dynamic_smem_bytes"] = capability.dynamic_smem_bytes;
    result["registers_per_thread"] = capability.registers_per_thread;
    result["max_threads_per_block"] = capability.max_threads_per_block;
    result["static_smem_bytes"] = capability.static_smem_bytes;
    result["max_dynamic_smem_bytes"] = capability.max_dynamic_smem_bytes;
    result["ptx_version"] = capability.ptx_version;
    result["binary_version"] = capability.binary_version;
    result["max_potential_cluster_size"] = capability.max_potential_cluster_size;
    result["max_active_clusters"] = capability.max_active_clusters;
    result["supported"] = capability.supported;
    result["reason"] = capability.reason;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
    m.def("topk_variant", &topk_variant);
    m.def("get_alignment_requirement", &get_alignment_requirement);
    m.def("get_cluster_capability", &get_cluster_capability);
}
