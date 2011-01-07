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
bool host_map_initialized = false;

thrust::host_vector<real>::iterator host_shell_x;
thrust::host_vector<real>::iterator host_shell_y;
thrust::host_vector<real>::iterator host_shell_z;

template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand_by_map_if(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed,
  thrust::host_space_tag
){
  //if(!host_map_initialized){
    THRUSTING_PP("===============initialized================", "");
    thrust::host_vector<real> xs(SHELL_TABLE_X, SHELL_TABLE_X+SHELL_RAND_MAP_SIZE);
    thrust::host_vector<real> ys(SHELL_TABLE_Y, SHELL_TABLE_Y+SHELL_RAND_MAP_SIZE);
    thrust::host_vector<real> zs(SHELL_TABLE_Z, SHELL_TABLE_Z+SHELL_RAND_MAP_SIZE);
    host_shell_x = xs.begin();   
    host_shell_y = ys.begin();   
    host_shell_z = zs.begin();   
    host_map_initialized = true;
  //}

  THRUSTING_PP("begin shell_rand_map_if", make_list(100, host_shell_z));

  thrusting::gather_if(
    n_particle,
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      compose(
        thrusting::make_uniform_int_distribution<size_t>(0, SHELL_RAND_MAP_SIZE), 
        thrusting::make_fast_rng_generator(seed))),
    stencil,
    thrusting::make_zip_iterator(
      host_shell_x,
      host_shell_y,
      host_shell_z),
    thrusting::make_zip_iterator(u, v, w),
    pred);
}

/*
  DEVICE PATH
*/
bool device_map_initialized = false;

thrust::device_vector<real>::iterator device_shell_x;
thrust::device_vector<real>::iterator device_shell_y;
thrust::device_vector<real>::iterator device_shell_z;

template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand_by_map_if(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed,
  thrust::device_space_tag
){
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
