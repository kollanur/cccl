// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <hpx/init.hpp>
#include "nvbench_helper.cuh"


using picoseconds = std::chrono::duration<long long, std::pico>;

struct adaptive_chunk_size {
  template <typename Rep1, typename Period1, typename Rep2, typename Period2>
  explicit constexpr adaptive_chunk_size(
    std::chrono::duration<Rep1, Period1> const time_per_iteration,
    std::chrono::duration<Rep2, Period2> const overhead_time = 
    std::chrono::microseconds(1)
  ) : time_per_iteration_(std::chrono::duration_cast<picoseconds>(time_per_iteration)), 
    overhead_time_(std::chrono::duration_cast<picoseconds>(overhead_time)) {
  }

  // calculate no of cores
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    hpx::execution::experimental::processing_units_count_t,
    adaptive_chunk_size& this_, Executor&& exec,
    hpx::chrono::steady_duration const&, std::size_t count
  ) noexcept {
    std::size_t const cores_baseline = 
     hpx::execution::experimental::processing_units_count(
      exec, this_.time_per_iteration_, count);
    
     auto const overall_time = static_cast<double>(
      (count + 1) * this_.time_per_iteration_.count());

    constexpr double efficiency_factor = 0.052;
    if(this_.overhead_time_.count()==0) {
      this_.overhead_time_ = std::chrono::duration_cast<picoseconds>(std::chrono::microseconds(1));
    }
    
    auto const optimal_num_cores = 
      static_cast<std::size_t>(efficiency_factor * overall_time / 
      static_cast<double>(this_.overhead_time_.count()));
    
    std::size_t num_cores = (std::min) (cores_baseline, optimal_num_cores);
    num_cores = (std::max) (num_cores, static_cast<std::size_t>(1)); 

    return num_cores;
  }

  //calculate chunk size
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    hpx::execution::experimental::get_chunk_size_t, adaptive_chunk_size&, 
    Executor&, hpx::chrono::steady_duration const&, std::size_t const cores,
    std::size_t num_iterations
  ) {
    if (cores == 1) {
      return num_iterations;
    }
    std::size_t times_cores = 8;
    
    if (cores ==2) {
      times_cores = 4;
    }

    // Return a chunk size that ensures that each core ends up with the same
    // number of chunks the sizes of which are equal (except for the last
    // chunk, which may be smaller by not more than the number of chunks in
    // terms of elements).

    std::size_t const num_chunks = times_cores * cores;
    std::size_t chunk_size = (num_iterations + num_chunks - 1) / num_chunks;

    // we should not consider more chunks than we have elements
    auto const max_chunks = (std::min) (num_chunks, num_iterations);

    // we should not make chunks smaller than what's determined by the max chunk size
    chunk_size = (std::max) (chunk_size, 
      (num_iterations + max_chunks -1)/ max_chunks);

    HPX_ASSERT(chunk_size * num_chunks >= num_iterations);

    return chunk_size;
  }

  picoseconds time_per_iteration_;
  picoseconds overhead_time_;
};

template <>
struct hpx::execution::experimental::is_executor_parameters<adaptive_chunk_size> : std::true_type {
};

template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename FwdIter3>
double run_adjacent_difference_benchmark_hpx(int const test_count, ExPolicy policy, FwdIter1 first, FwdIter2 last, FwdIter3 dest, BinaryOp op) {
  // warmup
  hpx::adjacent_difference(policy, first, last, dest, op);

  // actual measurement
  std::uint64_t time = hpx::chrono::high_resolution_clock::now();
  
  for(int i=0; i < test_count; ++i) {
     hpx::adjacent_difference(policy, first, last, dest, op);
  }

  time = hpx::chrono::high_resolution_clock::now() - time;

  return (static_cast<double>(time) * 1e-9) / test_count;
}

template <typename T>
struct custom_op
{
  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return std::pow(std::sin(std::tan(std::pow(lhs, 3)) *
                                     std::cos(std::pow(rhs, 3))),
                            5) +
                   std::pow(std::cos(std::tan(std::pow(rhs, 3)) *
                                     std::sin(std::pow(lhs, 3))),
                            5) +
                   std::exp(std::sqrt(
                       std::pow(std::sin(std::cos(lhs)) * std::cos(std::sin(rhs)),
                                6) +
                       std::pow(std::tan(std::exp(lhs * rhs)), 6))) +
                   std::log(std::abs(std::sin(std::exp(std::pow(lhs, 2))) *
                                         std::cos(std::exp(std::pow(rhs, 2))) +
                                     1e-9)) +
                   std::atan(std::pow(std::sin(std::exp(lhs + rhs)), 4)) *
                       std::acosh(std::pow(std::cos(std::exp(lhs * rhs)), 4)) +
                   std::pow(
                       std::hypot(std::log(std::abs(lhs)), std::log(std::abs(rhs))),
                       4) *
                       std::exp(std::log1p(std::pow(std::tan(lhs * rhs), 2)));
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  const double seq_time = run_adjacent_difference_benchmark_hpx(10, hpx::execution::seq, input.cbegin(), input.cend(), output.begin(), custom_op<T>{});
  double overhead_time = 0.0;
  double const time_per_iteration = seq_time / 
    static_cast<double>(elements);
  
  hpx::execution::experimental::num_cores nc(1);
  hpx::execution::experimental::max_num_chunks mnc(1);
  std::size_t const all_cores = hpx::get_num_worker_threads();
  
  auto temp = run_adjacent_difference_benchmark_hpx(10, hpx::execution::par.with(nc, mnc), input.cbegin(), input.cend(), output.begin(), custom_op<T>{});
  //std::cout<<"Sequential Time: "<<seq_time<<std::endl;
  //std::cout<<"parallel Time  : "<<temp<<std::endl;
  temp = temp - seq_time;
  //std::cout<<"Difference Time: "<<temp<<std::endl;
  overhead_time = (temp) / static_cast<double>(all_cores);
  
  picoseconds time_per_iteration_ps(
    static_cast<int64_t>(time_per_iteration * 1e12)
  );
  picoseconds overhead_time_ps(
    static_cast<int64_t>(overhead_time * 1e12)
  );

  adaptive_chunk_size acs(time_per_iteration_ps, overhead_time_ps);
  hpx::execution::experimental::chunking_parameters params = {};
  hpx::execution::experimental::collect_chunking_parameters
               collect_params(params);


  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::adjacent_difference(
                 policy(alloc, launch).with(acs, collect_params), input.cbegin(), input.cend(), output.begin(), custom_op<T>{});
             });

std::cout<<"Chunk size: "<<params.chunk_size<<" No of chunks: "<<params.num_chunks<<" No of cores: "<<params.num_cores<<std::endl;
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(8, 30, 1));
