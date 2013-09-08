#pragma once

#include <thrust/transform.h>
#include <thrust/iterator/permutation_iterator.h>

#include <thrusting/functional.h>
#include <thrusting/algorithm/reduce_by_bucket.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/real.h>
#include <thrusting/vectorspace.h>

#include <iostream>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<typename Real, typename Int>
void minus_average_velocity(
  size_t n_particle,
  Real u, Real v, Real w, // input and output
  Int idx, // input
  size_t n_cell,
  Real ave_u, Real ave_v, Real ave_w, // output
  Real tmp3, Real tmp4, Real tmp5,
  Int tmp1, Int tmp2
){
  /*
    calc sum velocity each cell
  */
  real3 zero_veloc(0.0, 0.0, 0.0);
  thrusting::reduce_by_bucket(
    n_particle,
    idx,
    thrusting::make_zip_iterator(u, v, w),
    n_cell,
    tmp1,
    tmp2, // cnt
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
    thrusting::make_zip_iterator(tmp3, tmp4, tmp5),
    zero_veloc);
    
  Int cnt = tmp2;
  /*
    average the velocity
    if cnt = 0 then no calc
  */
  thrust::transform_if(
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
    thrusting::advance(n_cell, thrusting::make_zip_iterator(ave_u, ave_v, ave_w)),
    cnt, // input2
    cnt, // stencil
    thrusting::make_zip_iterator(ave_u, ave_v, ave_w), // output
    thrusting::divides<real3, real>(), // if not 0 divides
    thrusting::bind2nd(thrust::not_equal_to<size_t>(), 0)); 
     
  /*
    minus 
  */
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrust::make_permutation_iterator(
      thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
      idx),
    thrusting::make_zip_iterator(u, v, w),
    thrust::minus<real3>());
}
  
template<typename Real, typename Int>
void plus_average_velocity(
  size_t n_particle,
  Real u, Real v, Real w, // input and output
  Int idx,
  size_t n_cell,
  Real ave_u, Real ave_v, Real ave_w // input
){
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrust::make_permutation_iterator(
      thrusting::make_zip_iterator(ave_u, ave_v, ave_w),
      idx),
    thrusting::make_zip_iterator(u, v, w),
    thrust::plus<real3>());
}

} // END bphcuda
