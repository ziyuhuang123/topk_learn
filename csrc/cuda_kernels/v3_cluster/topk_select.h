#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime_api.h>

#include "structs.h"

namespace topk_select_bf16_cluster {

struct ClusterCapability {
    uint32_t requested_cluster_size = 0;
    size_t dynamic_smem_bytes = 0;
    int registers_per_thread = -1;
    int max_threads_per_block = -1;
    size_t static_smem_bytes = 0;
    int max_dynamic_smem_bytes = -1;
    int ptx_version = -1;
    int binary_version = -1;
    int max_potential_cluster_size = 0;
    int max_active_clusters = 0;
    bool supported = false;
    std::string reason;
};

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args);

template<typename Config>
ClusterCapability get_cluster_capability(cudaStream_t stream);

}  // namespace topk_select_bf16_cluster
