#pragma once

#include <thrusting/algorithm/gather.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/random/engine.h>
#include <thrusting/random/distribution.h>
#include <thrusting/vector.h>
#include <thrusting/pp.h>
#include <thrusting/list.h>

#include <thrust/iterator/counting_iterator.h>

#include "shell_rand_map.h"

namespace {
  using namespace thrust;
  using namespace thrusting;
}

namespace bphcuda {
namespace detail {

/*
  HOST PATH
*/
thrust::host_vector<real> host_x(SHELL_TABLE_X, SHELL_TABLE_X+SHELL_RAND_MAP_SIZE);
thrust::host_vector<real> host_y(SHELL_TABLE_Y, SHELL_TABLE_Y+SHELL_RAND_MAP_SIZE);
thrust::host_vector<real> host_z(SHELL_TABLE_Z, SHELL_TABLE_Z+SHELL_RAND_MAP_SIZE);

template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand_by_map_if(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed,
  thrust::host_space_tag
){
  thrusting::gather_if(
    n_particle,
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      compose(
        thrusting::make_uniform_int_distribution<size_t>(0, SHELL_RAND_MAP_SIZE), 
        thrusting::make_fast_rng_generator(seed))),
    stencil,
    thrusting::make_zip_iterator(
      host_x.begin(),
      host_y.begin(),
      host_z.begin()),
    thrusting::make_zip_iterator(u, v, w),
    pred);
}

/*
  DEVICE PATH
*/
thrust::device_vector<real> device_x(SHELL_TABLE_X, SHELL_TABLE_X+SHELL_RAND_MAP_SIZE);
thrust::device_vector<real> device_y(SHELL_TABLE_Y, SHELL_TABLE_Y+SHELL_RAND_MAP_SIZE);
thrust::device_vector<real> device_z(SHELL_TABLE_Z, SHELL_TABLE_Z+SHELL_RAND_MAP_SIZE);

template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand_by_map_if(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed,
  thrust::device_space_tag
){
  thrusting::gather_if(
    n_particle,
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      compose(
        thrusting::make_uniform_int_distribution<size_t>(0, SHELL_RAND_MAP_SIZE), 
        thrusting::make_fast_rng_generator(seed))),
    stencil,
    thrusting::make_zip_iterator(
      device_x.begin(),
      device_y.begin(),
      device_z.begin()),
    thrusting::make_zip_iterator(u, v, w),
    pred);
}

} // END detail

/*
  Dispatcher
*/
template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand_by_map_if(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed
){
  THRUSTING_PP("before alloc_shell_rand_map_if : u", make_list(5, u));
  THRUSTING_PP("before alloc_shell_rand_map_if : v", make_list(5, v));
  THRUSTING_PP("before alloc_shell_rand_map_if : w", make_list(5, w));
  detail::alloc_shell_rand_by_map_if(
    n_particle,
    u, v, w,
    stencil,
    pred,
    seed,
    typename thrust::iterator_space<Real>::type());
  THRUSTING_PP("after alloc_shell_rand_map_if : u", make_list(5, u));
  THRUSTING_PP("after alloc_shell_rand_map_if : v", make_list(5, v));
  THRUSTING_PP("after alloc_shell_rand_map_if : w", make_list(5, w));
}

} // END bphcuda
