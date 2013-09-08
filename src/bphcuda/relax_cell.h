#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution/shell_distribution.h>

#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
template<typename Real>
void alloc_new_c(
  size_t n_particle,
  Real u, Real v, Real w,
  size_t seed
){
  /*
    First allocate a zero vector at the end of given vector
  */
  thrusting::alloc_at(n_particle-1, thrusting::make_zip_iterator(u, v, w), real3(0.0, 0.0, 0.0));
  size_t h_len = n_particle / 2; 
  /*
    Alloc shell random to the first half
  */
  alloc_shell_rand(h_len, u, v, w, seed);
  thrust::copy(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)));
  /*
    Alloc its inverse to the second half
  */
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)), 
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), real(-1.0)));
}
} // END detail

template<typename Real>
void relax_cell (
  size_t n_particle,
  Real u, Real v, Real w,
  real m,
  Real in_e,
  real s,
  size_t seed
){
  /*
    if n_particle is less than 2 then
    relax will not occur because there is theoretically no particle collision.
  */
  if(n_particle < 2){ return; }

  thrust::constant_iterator<real> m_it(m);

  real old_total_e = calc_total_e(n_particle, u, v, w, m_it, in_e); 
  
  real new_total_kinetic_e = real(3) / (3+s) * old_total_e;
  real new_total_in_e = s / (3+s) * old_total_e;

  detail::alloc_new_c(
    n_particle,
    u, v, w,
    seed);   

  real tmp_total_kinetic_e = bphcuda::calc_kinetic_e(
    n_particle,
    u, v, w,
    m_it);
   
  real ratio_c = sqrt(new_total_kinetic_e / tmp_total_kinetic_e);
  
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), real(ratio_c)));
  
  thrust::fill(
    in_e,
    thrusting::advance(n_particle, in_e),
    real(new_total_in_e / n_particle)); 
}

} // END bphcuda
