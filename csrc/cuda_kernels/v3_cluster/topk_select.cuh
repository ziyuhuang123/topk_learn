/*
TopK select BF16 cluster variant. Multiple CTAs scan one row, gather their local
candidates into rank 0 through DSM, and rank 0 performs the final selection.
*/

#pragma once

#include "topk_select.h"

#include <climits>
#include <string>

#include <cutlass/kernel_launch.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_bf16_cluster {

CUTE_DEVICE
static void cluster_rendezvous() {
    ku::barrier_cluster_arrive_release();
    ku::barrier_cluster_wait_acquire();
}

CUTE_DEVICE
static void st_async_32b(uint32_t dst_addr, uint32_t data, uint32_t mbar_addr) {
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.s32 [%0], {%1}, [%2];\n"
        :
        : "r"(dst_addr), "r"(data), "r"(mbar_addr)
        : "memory"
    );
}

template<typename T>
CUTE_DEVICE
static void st_async_128b(uint32_t dst_addr, const T &data, uint32_t mbar_addr) {
    static_assert(sizeof(T) == 16, "DSM async stores must be 128 bits");
    long2 data_long2 = *reinterpret_cast<const long2*>(&data);
    asm volatile(
        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64 [%0], {%1, %2}, [%3];\n"
        :
        : "r"(dst_addr), "l"(data_long2.x), "l"(data_long2.y), "r"(mbar_addr)
        : "memory"
    );
}

template<typename Config>
class TopkSelectKernelBF16Cluster : public topk_select_common::TopkSelectKernelBF16Base<Config> {
    using BF16Base = topk_select_common::TopkSelectKernelBF16Base<Config>;

public:
    using ValueT = typename BF16Base::ValueT;
    using OutIdxT = typename BF16Base::OutIdxT;
    using TmaParams = typename BF16Base::TmaParams;
    using SharedMemoryPlanBase = typename BF16Base::SharedMemoryPlanBase;
    using EpilogueT = typename BF16Base::EpilogueT;
    using BF16Base::NUM_THREADS;
    using BF16Base::MAX_TOPK;
    using BF16Base::NUM_ELEMS_PER_SEG;
    using BF16Base::NUM_ELEMS_PER_ROUND;
    using BF16Base::NUM_SEGS_IN_INIT_WINDOW;
    using BF16Base::NUM_INIT_ROUNDS_MAX;
    using BF16Base::NUM_TAIL_ELEMS;
    using BF16Base::NUM_TAIL_SEGS;
    using BF16Base::NUM_SEGS_PER_ROUND;
    using BF16Base::PERM_ADD_BASE;

    static constexpr uint32_t CLUSTER_SIZE = Config::cluster_size;
    static constexpr uint32_t NUM_GATHER_PAIRS = CLUSTER_SIZE * MAX_TOPK;
    static constexpr uint32_t NUM_GATHER_UNITS = NUM_GATHER_PAIRS / 2;
    static constexpr uint32_t NUM_UINT32_GATHER_PER_THREAD =
        ((NUM_GATHER_UNITS + NUM_THREADS - 1) / NUM_THREADS) | 1u;

    static_assert(cute::is_same_v<ValueT, nv_bfloat16>);
    static_assert(cute::is_same_v<OutIdxT, int64_t>);
    static_assert(!Config::sorted_value && !Config::sorted_index && Config::return_value);
    static_assert(MAX_TOPK == 512);
    static_assert(NUM_THREADS == 256);
    static_assert(Config::target_occupancy == 1);
    static_assert(NUM_ELEMS_PER_ROUND == 4096);
    static_assert(Config::reconstruct_threshold == 4096);
    static_assert(Config::tma_buffer_depth == CLUSTER_SIZE);
    static_assert(Config::candidate_compaction_policy == CandidateCompactionPolicy::Ballot);
    static_assert(Config::tma_schedule_policy == TmaSchedulePolicy::Pipelined);
    static_assert(Config::reconstruct_policy == ReconstructPolicy::Adaptive);
    static_assert(CLUSTER_SIZE == 2 || CLUSTER_SIZE == 4 || CLUSTER_SIZE == 8,
                  "cluster_size must be 2, 4, or 8");
    static_assert(CLUSTER_SIZE <= 8, "only portable cluster sizes are supported");
    static_assert(NUM_GATHER_PAIRS % 2 == 0);
    static_assert(NUM_GATHER_UNITS % NUM_THREADS == 0,
                  "gather units must divide evenly across threads");
    static_assert(NUM_GATHER_PAIRS <= 0xFFFF);
    static_assert(2 * NUM_UINT32_GATHER_PER_THREAD <= 256);
    static_assert(NUM_ELEMS_PER_ROUND * sizeof(ValueT) % 1024 == 0);
    static_assert((uint64_t)(MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG) *
                          (MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG) +
                      PERM_ADD_BASE <=
                  0xFFFFFFFFull);

    struct SharedMemoryPlanBF16Cluster : SharedMemoryPlanBase {
        kerutils::transac_bar_t gather_val_bar;
        kerutils::transac_bar_t gather_bar;
        CUTE_ALIGNAS(16) uint32_t gathered_num_survivors[CLUSTER_SIZE];
    };

    static_assert(NUM_GATHER_UNITS * sizeof(uint32_t) <=
                      sizeof(SharedMemoryPlanBase::incoming_topk_pairs),
                  "gathered values must fit in incoming_topk_pairs");
    static_assert(NUM_GATHER_PAIRS * sizeof(uint64_t) <=
                      sizeof(SharedMemoryPlanBase::tma_load_buf),
                  "gathered pairs must fit in the configured TMA buffers");

    static __device__ __forceinline__
    void topk_select_kernel_devfunc(const TopkSelectArgs &args, const TmaParams &tma_params) {
        const uint32_t rank_in_cluster = cute::block_rank_in_cluster();
        const uint32_t batch_idx = blockIdx.y;
        const uint32_t end_vocab_idx =
            args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlanBF16Cluster &smem =
            *reinterpret_cast<SharedMemoryPlanBF16Cluster*>(wksp_buf);

        const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
        const uint32_t lane_idx = threadIdx.x % 32;

        // This branch is uniform across the cluster because every rank has the same batch_idx.
        // Rank 0 writes the shortcut result, then every rank participates in the final rendezvous.
        if (end_vocab_idx <= args.topk) {
            if (rank_in_cluster == 0) {
                EpilogueT::template topk_select_epilogue<true>(
                    reinterpret_cast<ValueT*>(smem.surviving_topk_pairs[0]),
                    reinterpret_cast<uint32_t*>(smem.surviving_topk_pairs[1]),
                    args,
                    batch_idx,
                    end_vocab_idx,
                    warp_idx,
                    *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(
                        smem.incoming_topk_pairs)
                );
            }
            cluster_rendezvous();
            return;
        }

        uint32_t survivor_buf_idx = 0;
        bool nan_seen = false;

        BF16Base::init_shared_memory(smem, warp_idx, [&] {
            smem.gather_val_bar.init(1);
            smem.gather_bar.init(1);
        });

        // Compose fence.mbarrier_init.release.cluster from init_shared_memory with a
        // cluster-scoped rendezvous before any rank can target rank 0's barriers.
        cluster_rendezvous();

        const uint32_t num_input_segs =
            ku::ceil_div(end_vocab_idx, static_cast<uint32_t>(NUM_ELEMS_PER_SEG));
        const uint32_t num_perm_segs =
            num_input_segs <= NUM_SEGS_IN_INIT_WINDOW
                ? 0u
                : (num_input_segs - 1) / NUM_TAIL_SEGS * NUM_TAIL_SEGS;
        const uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
        const uint32_t num_tail_elems_padded =
            num_perm_segs != 0 ? static_cast<uint32_t>(NUM_TAIL_ELEMS)
                               : num_input_segs * NUM_ELEMS_PER_SEG;

        // Rank 0 owns the tail. The permuted prefix is divided into contiguous ranges
        // in visit-order space so all ranks receive approximately equal work.
        const uint32_t num_local_tail_elems_padded =
            rank_in_cluster == 0 ? num_tail_elems_padded : 0u;
        const uint32_t num_local_tail_segs =
            num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
        const uint32_t num_local_tail_elems =
            rank_in_cluster == 0 ? end_vocab_idx - num_perm_elems : 0u;
        const uint32_t num_total_elems_padded = num_tail_elems_padded + num_perm_elems;

        auto perm_boundary = [&](uint32_t rank) {
            static_assert((uint64_t)CLUSTER_SIZE * (MAX_VOCAB_SIZE + NUM_TAIL_ELEMS) <=
                          0xFFFFFFFFull);
            const uint32_t boundary = rank * num_total_elems_padded / CLUSTER_SIZE;
            return boundary > num_tail_elems_padded ? boundary - num_tail_elems_padded : 0u;
        };

        const uint32_t local_start_seg_idx =
            perm_boundary(rank_in_cluster) / NUM_ELEMS_PER_SEG;
        const uint32_t local_end_seg_idx =
            perm_boundary(rank_in_cluster + 1) / NUM_ELEMS_PER_SEG;
        const uint32_t num_local_perm_segs = local_end_seg_idx - local_start_seg_idx;
        const uint32_t num_local_elems =
            num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;
        const uint32_t num_local_rounds =
            ku::ceil_div(num_local_elems, static_cast<uint32_t>(NUM_ELEMS_PER_ROUND));
        const uint32_t num_local_init_rounds =
            min(num_local_rounds, static_cast<uint32_t>(NUM_INIT_ROUNDS_MAX));
        const uint32_t num_local_segs = num_local_tail_segs + num_local_perm_segs;
        const uint32_t used_segs = num_local_init_rounds * NUM_SEGS_PER_ROUND;
        const uint32_t rem_segs =
            num_local_segs > used_segs ? num_local_segs - used_segs : 0u;
        const uint32_t warp_rem_segs =
            warp_idx < rem_segs ? rem_segs - warp_idx : 0u;
        const uint32_t warp_round_limit =
            ku::ceil_div(warp_rem_segs, static_cast<uint32_t>(NUM_SEGS_PER_ROUND));

        const uint32_t topk_len = BF16Base::template scan_segs<true>(
            tma_params,
            smem,
            batch_idx,
            end_vocab_idx,
            args.topk,
            warp_idx,
            lane_idx,
            num_perm_segs,
            local_start_seg_idx,
            num_local_perm_segs,
            num_local_tail_elems_padded,
            num_local_tail_elems,
            survivor_buf_idx,
            nan_seen,
            [&](uint32_t round) { return round < warp_round_limit; }
        );

        nan_seen = __syncthreads_or(nan_seen) != 0;

        // No rank may overwrite rank 0's scan scratch until rank 0 has finished using it.
        cluster_rendezvous();

        const uint32_t val_dst = cute::set_block_rank(
            cute::cast_smem_ptr_to_uint(
                reinterpret_cast<uint32_t*>(smem.incoming_topk_pairs) +
                rank_in_cluster * (MAX_TOPK / 2)),
            0);
        const uint32_t pair_dst = cute::set_block_rank(
            cute::cast_smem_ptr_to_uint(
                reinterpret_cast<uint64_t*>(smem.tma_load_buf) +
                rank_in_cluster * MAX_TOPK),
            0);
        const uint32_t gather_val_bar_addr = cute::set_block_rank(
            cute::cast_smem_ptr_to_uint(&smem.gather_val_bar), 0);
        const uint32_t gather_bar_addr = cute::set_block_rank(
            cute::cast_smem_ptr_to_uint(&smem.gather_bar), 0);

        static_assert(MAX_TOPK * sizeof(ValueT) % 16 == 0);
        constexpr uint32_t NUM_VAL_CHUNKS = MAX_TOPK * sizeof(ValueT) / 16;
        constexpr uint32_t NUM_VAL_CHUNKS_PER_THREAD =
            ku::ceil_div(NUM_VAL_CHUNKS, NUM_THREADS);

        // Rank 0 registers all expected remote bytes before any rank issues a DSM store.
        if (warp_idx == 0 && cute::elect_one_sync() && rank_in_cluster == 0) {
            smem.gather_val_bar.arrive_and_expect_tx(
                (NUM_VAL_CHUNKS * 16 + sizeof(uint32_t)) * CLUSTER_SIZE);
            smem.gather_bar.arrive_and_expect_tx(
                (MAX_TOPK * static_cast<uint32_t>(sizeof(uint64_t))) * CLUSTER_SIZE);
        }
        __syncthreads();
        cluster_rendezvous();

        if (warp_idx == 0 && cute::elect_one_sync()) {
            const uint32_t survivor_count_dst = cute::set_block_rank(
                cute::cast_smem_ptr_to_uint(
                    smem.gathered_num_survivors + rank_in_cluster),
                0);
            st_async_32b(
                survivor_count_dst,
                topk_len | (static_cast<uint32_t>(nan_seen) << 31),
                gather_val_bar_addr);
        }

        // Gather values first so rank 0 can begin pivot selection while pairs arrive.
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_VAL_CHUNKS_PER_THREAD; ++i) {
            const uint32_t chunk = i * NUM_THREADS + threadIdx.x;
            if constexpr (NUM_VAL_CHUNKS % NUM_THREADS != 0) {
                if (chunk >= NUM_VAL_CHUNKS) {
                    break;
                }
            }
            uint32_t packed_values[4];
            CUTE_UNROLL
            for (uint32_t j = 0; j < 4; ++j) {
                uint32_t pair2[4];
                topk_select_common::ld_shared<4>(
                    pair2,
                    reinterpret_cast<const uint32_t*>(
                        smem.surviving_topk_pairs[survivor_buf_idx] +
                        8 * chunk + 2 * j));
                packed_values[j] = __byte_perm(pair2[1], pair2[3], 0x5410);
            }
            st_async_128b(
                val_dst + chunk * sizeof(uint4),
                make_uint4(
                    packed_values[0], packed_values[1], packed_values[2], packed_values[3]),
                gather_val_bar_addr);
        }

        static_assert(MAX_TOPK * sizeof(uint64_t) % 16 == 0);
        constexpr uint32_t NUM_PAIR_CHUNKS = MAX_TOPK * sizeof(uint64_t) / 16;
        static_assert(NUM_PAIR_CHUNKS % NUM_THREADS == 0,
                      "pair gather chunks must divide evenly across threads");
        CUTE_UNROLL
        for (uint32_t i = 0; i < NUM_PAIR_CHUNKS / NUM_THREADS; ++i) {
            const uint32_t pair = 2 * (i * NUM_THREADS + threadIdx.x);
            const ulonglong2 packed_pairs = *reinterpret_cast<const ulonglong2*>(
                smem.surviving_topk_pairs[survivor_buf_idx] + pair);
            st_async_128b(
                pair_dst + pair * sizeof(uint64_t), packed_pairs, gather_bar_addr);
        }

        if (rank_in_cluster == 0) {
            smem.gather_val_bar.wait(0);
            BF16Base::clear_reconstruct_histograms(smem, threadIdx.x);
            __syncthreads();

            const uint32_t stored_num_survivors =
                lane_idx < CLUSTER_SIZE ? smem.gathered_num_survivors[lane_idx] : 0u;
            const uint32_t sum_topk_len =
                __reduce_add_sync(0xFFFFFFFFu, stored_num_survivors & 0x7FFFFFFFu);
            nan_seen |= (stored_num_survivors >> 31) != 0;

            const uint32_t num_nan_pads = NUM_GATHER_PAIRS - sum_topk_len;
            const uint32_t unit_base = threadIdx.x * NUM_UINT32_GATHER_PER_THREAD;
            const uint32_t num_my_units =
                unit_base < NUM_GATHER_UNITS
                    ? min(static_cast<uint32_t>(NUM_UINT32_GATHER_PER_THREAD),
                          NUM_GATHER_UNITS - unit_base)
                    : 0u;

            const uint32_t *gather_vals =
                reinterpret_cast<const uint32_t*>(smem.incoming_topk_pairs);
            nv_bfloat162 values[NUM_UINT32_GATHER_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_UINT32_GATHER_PER_THREAD; ++i) {
                if (i == num_my_units) {
                    break;
                }
                values[i] = topk_select_common::u32_to_bf16x2(
                    gather_vals[unit_base + i]);
            }
            BF16Base::histogram_radix_msb(
                smem.reconstruct_bucket_counter[0], values, num_my_units);
            __syncthreads();

            auto [pivot_value_x2_bits, out_prefix, eq_quota, cnt_nan] =
                BF16Base::template compute_pivot_and_quota<true>(
                    args.topk,
                    num_nan_pads,
                    values,
                    num_my_units,
                    warp_idx,
                    lane_idx,
                    smem);
            (void)cnt_nan;

            smem.gather_bar.wait(0);
            __syncthreads();

            uint32_t out_ptr = cute::cast_smem_ptr_to_uint(
                                   smem.surviving_topk_pairs[survivor_buf_idx ^ 1]) +
                               out_prefix * static_cast<uint32_t>(sizeof(uint64_t));
            const uint32_t gather_pairs_base =
                cute::cast_smem_ptr_to_uint(smem.tma_load_buf);
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_UINT32_GATHER_PER_THREAD; ++i) {
                if (i == num_my_units) {
                    break;
                }
                BF16Base::template copy_selected_pairs_to_survivor<true>(
                    out_ptr,
                    eq_quota,
                    values[i],
                    pivot_value_x2_bits,
                    gather_pairs_base +
                        2 * (unit_base + i) * static_cast<uint32_t>(sizeof(uint64_t)));
            }
            survivor_buf_idx ^= 1;

            nan_seen = __syncthreads_or(nan_seen) != 0;
            if (nan_seen) {
                BF16Base::take_action_when_have_nan(args, batch_idx);
            } else {
                BF16Base::template stage_output_and_epilogue<false>(
                    smem,
                    *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(
                        smem.incoming_topk_pairs),
                    args,
                    batch_idx,
                    end_vocab_idx,
                    survivor_buf_idx,
                    warp_idx,
                    lane_idx);
            }
        }

        // Rank 0 reaches this only after both gather barriers have completed and output
        // no longer uses DSM-backed scratch. Other ranks cannot retire before then.
        cluster_rendezvous();
    }
};

template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, Kernel::CLUSTER_SIZE)
__global__ void topk_kernel(
    __grid_constant__ const TopkSelectArgs args,
    __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::topk_select_kernel_devfunc(args, tma_params);
}

template<typename Kernel>
static ClusterCapability query_cluster_capability(cudaStream_t stream) {
    ClusterCapability capability;
    capability.requested_cluster_size = Kernel::CLUSTER_SIZE;
    capability.dynamic_smem_bytes = sizeof(typename Kernel::SharedMemoryPlanBF16Cluster);

    auto kernel = topk_kernel<Kernel>;
    auto fail = [&](const char *operation, cudaError_t status) {
        capability.reason = std::string(operation) + ": " + cudaGetErrorString(status);
        capability.supported = false;
        return capability;
    };

    static_assert(sizeof(typename Kernel::SharedMemoryPlanBF16Cluster) <= INT_MAX);
    cudaError_t status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(capability.dynamic_smem_bytes));
    if (status != cudaSuccess) {
        return fail("cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed", status);
    }

    cudaFuncAttributes function_attributes{};
    status = cudaFuncGetAttributes(&function_attributes, kernel);
    if (status != cudaSuccess) {
        return fail("cudaFuncGetAttributes failed", status);
    }
    capability.registers_per_thread = function_attributes.numRegs;
    capability.max_threads_per_block = function_attributes.maxThreadsPerBlock;
    capability.static_smem_bytes = function_attributes.sharedSizeBytes;
    capability.max_dynamic_smem_bytes = function_attributes.maxDynamicSharedSizeBytes;
    capability.ptx_version = function_attributes.ptxVersion;
    capability.binary_version = function_attributes.binaryVersion;

    cudaLaunchAttribute launch_attributes[2]{};
    launch_attributes[0].id = cudaLaunchAttributeClusterDimension;
    launch_attributes[0].val.clusterDim = {Kernel::CLUSTER_SIZE, 1, 1};
    launch_attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    launch_attributes[1].val.programmaticStreamSerializationAllowed = 0;

    cudaLaunchConfig_t launch_config = {
        dim3(Kernel::CLUSTER_SIZE, 1, 1),
        dim3(Kernel::NUM_THREADS, 1, 1),
        capability.dynamic_smem_bytes,
        stream,
        launch_attributes,
        2
    };

    status = cudaOccupancyMaxPotentialClusterSize(
        &capability.max_potential_cluster_size, kernel, &launch_config);
    if (status != cudaSuccess) {
        return fail("cudaOccupancyMaxPotentialClusterSize failed", status);
    }

    status = cudaOccupancyMaxActiveClusters(
        &capability.max_active_clusters, kernel, &launch_config);
    if (status != cudaSuccess) {
        return fail("cudaOccupancyMaxActiveClusters failed", status);
    }

    if (capability.max_potential_cluster_size < static_cast<int>(Kernel::CLUSTER_SIZE)) {
        capability.reason = "requested cluster size exceeds the kernel/device maximum";
    } else if (capability.max_active_clusters <= 0) {
        capability.reason = "no active cluster can be scheduled for this specialization";
    } else {
        capability.supported = true;
    }
    return capability;
}

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args) {
    KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
    KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
    KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
    KU_ASSERT(!args.abort_when_nan_found,
              "Cluster benchmark variants require abort_when_nan_found=False");
    KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

    using Kernel = TopkSelectKernelBF16Cluster<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK,
              "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlanBF16Cluster);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)));

    KU_ASSERT(args.stride_input_batch % 8 == 0,
              "stride_input_batch must be 16B-aligned");
    typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

    ku::launch_kernel(
        ku::KernelLaunchConfig{
            dim3(Kernel::CLUSTER_SIZE, args.batch_size, 1),
            dim3(Kernel::NUM_THREADS, 1, 1),
            smem_size,
            args.stream,
            dim3(Kernel::CLUSTER_SIZE, 1, 1)},
        kernel,
        args,
        tma_params);
}

template<typename Config>
ClusterCapability get_cluster_capability(cudaStream_t stream) {
    using Kernel = TopkSelectKernelBF16Cluster<Config>;
    return query_cluster_capability<Kernel>(stream);
}

}  // namespace topk_select_bf16_cluster
