#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#include <thrusting/iterator.h>
#include <thrusting/real.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/functional.h>
#include <thrusting/algorithm/reduce_by_bucket.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<typename Real, typename Int>
void alloc_in_e(
  size_t n_particle, 
  Real u, Real v, Real w,
  real m,
  Real in_e, // output
  real s,
  Int idx,
  size_t n_cell,
  Real tmp1, Real tmp4,
  Int tmp2, Int tmp3
){
  thrust::fill(
    in_e,
    thrusting::advance(n_particle, in_e),
    real(0));
   
  thrust::constant_iterator<real> m_it(m);
  thrusting::reduce_by_bucket(
    n_particle,
    idx,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w, m_it),
      make_kinetic_e_calculator()),
    thrust::plus<real>(),
    n_cell,
    tmp2, 
    tmp3, // cnt
    tmp1, // sum of e_kin by cell
    tmp4,
    real(0)); // default e is 0
  
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp1, // sum of e_in by cell
    thrusting::bind2nd(thrust::multiplies<real>(), s/3)); 
   
  thrust::transform_if(
    tmp1, thrusting::advance(n_cell, tmp1), // input1
    tmp3, // input2, n particle in cell
    tmp3, // stencil, cnt
    tmp1, // result
    thrust::divides<real>(), // op 
    thrusting::bind2nd(thrust::not_equal_to<size_t>(), 0));

  thrust::gather(
    idx, 
    thrusting::advance(n_particle, idx),
    tmp1, // value
    in_e); 
}

} // END bphcuda
