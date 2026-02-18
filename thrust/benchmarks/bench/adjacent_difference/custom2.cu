// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>

#include "nvbench_helper.cuh"

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

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::adjacent_difference(
                 policy(alloc, launch), input.cbegin(), input.cend(), output.begin(), custom_op<T>{});
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 18, 1));
