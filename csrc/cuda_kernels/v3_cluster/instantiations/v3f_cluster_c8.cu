#include "../topk_select.cuh"

namespace topk_select_bf16_cluster {

using ClusterC8Config = TopkSelectConfig<
    nv_bfloat16, int64_t, false, false, true,
    512, 256, 1, 4096, 4096, 8, 512, 8>;

template
void run_topk_select_kernel<ClusterC8Config>(const TopkSelectArgs &args);

template
ClusterCapability get_cluster_capability<ClusterC8Config>(cudaStream_t stream);

}  // namespace topk_select_bf16_cluster
