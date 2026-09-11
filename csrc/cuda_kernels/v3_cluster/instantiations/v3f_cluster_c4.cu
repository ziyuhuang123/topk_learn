#include "../topk_select.cuh"

namespace topk_select_bf16_cluster {

using ClusterC4Config = TopkSelectConfig<
    nv_bfloat16, int64_t, false, false, true,
    512, 256, 1, 4096, 4096, 4, 512, 4>;

template
void run_topk_select_kernel<ClusterC4Config>(const TopkSelectArgs &args);

template
ClusterCapability get_cluster_capability<ClusterC4Config>(cudaStream_t stream);

}  // namespace topk_select_bf16_cluster
