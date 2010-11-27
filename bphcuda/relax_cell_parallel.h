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

/*
  [real3] -> [real3]
  allocate new velocity to particle
  sustaining the momentum.
  This function assumes every particles are same in mass.
*/
template<typename Real>
void alloc_new_c(
  size_t n_particle,
  Real u, Real v, Real w,
  size_t seed
){
  // First allocate a zero vector at the end of given vector
  thrusting::alloc_at(n_particle-1, thrusting::make_zip_iterator(u, v, w), real3(0.0, 0.0, 0.0));
  size_t h_len = n_particle / 2; 
  alloc_shell_rand(h_len, u, v, w, seed);
  thrust::copy(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)));
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)), 
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), -1.0));
}

/*
  c -> c

  Algorithm,
  1. allocate new velocity sustaining the momentum.
  2. scaling the velocity to recover the total kinetic energy in prior to this procedure.
  This function assumes every particles are same in mass.
*/
template<typename Real>
void relax_kinetic_e(
  size_t n_particle,
  Real u, Real v, Real w,
  size_t seed
){
  if(n_particle < 2) { return; }
  const real some_m = 1.0;
  real old_kinetic_e = calc_kinetic_e(n_particle, u, v, w, thrust::constant_iterator<real>(some_m));
  alloc_new_c(n_particle, u, v, w, seed);
  real new_kinetic_e = calc_kinetic_e(n_particle, u, v, w, thrust::constant_iterator<real>(some_m));
  real ratio = sqrt(old_kinetic_e / new_kinetic_e);
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), ratio));
}

/*
*/
template<typename Real>
void relax_cell_parallel(
  size_t n_particle,
  Real u, Real v, Real w,
  real m,
  Real in_e,
  real s,
  size_t seed
){
  relax_kinetic_e(
    n_particle, 
    u, v, w,
    seed);

  constant_iterator<real> m_it = make_constant_iterator(m);
  constant_iterator<real> s_it = make_constant_iterator(s);
  
  thrust::transfrom(
    thrusting::make_zip_iterator(u, v, w, m_it, s_it),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m_it, s_it)),
    in_e,
    share_e_function()); 
}

}
