#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <bphcuda/distribution.h>
#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
}

namespace bphcuda {

namespace {
/*
  [real3] -> [real3]
  allocate new velocity to particle
  sustaining the momentum.
  This function assumes every particles are same in mass.
*/
template<typename RealIteartor>
void alloc_new_c(
  size_t n_particle,
  RealIteartor u, RealIteartor v, RealIteartor w,
  size_t seed
){
  // First allocate a zero vector at the end of given vector
  alloc_at(n_particle-1, make_zip_iterator(u, v, w), real3(0.0, 0.0, 0.0));
  size_t h_len = n_particle / 2; 
  alloc_shell_rand(h_len, u, v, w, seed):
  thrust::copy(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)));
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(h_len, thrusting::make_zip_iterator(u, v, w)), 
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(-1.0)));
}
} // END namespace
  
// Future
//template<typename RealIterator>
//void alloc_new_c(
//  size_t n_particle,
//  RealIterator u, RealIterator v, RealIterator w,
//  RealIterator m,
//  size_t seed
//){
//}

/*
  [real3] -> [real3]
  Algorithm,
  1. allocate new velocity sustaining the momentum.
  2. scaling the velocity to recover the total kinetic energy in prior to this procedure.
  This function assumes every particles are same in mass.
*/
template<typename RealIterator>
void relax(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  size_t seed
){
  if(n_particle < 2) { return; }
  const real some_m = 1.0;
  real old_kinetic_e = calc_kinetic_e(n_particle, u, v, w, thrust::const_iterator<real>(some_m));
  alloc_new_c(n_particle, u, v, w, seed);
  real new_kinetic_e = calc_kinetic_e(n_particle, u, v, w, thrust::const_iterator<real>(some_m));
  real ratio = sqrt(old_kinetic_e / new_kinetic_e);
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(ratio)));
}

// Future
//template<typename RealIterator>
//void relax(
//  size_t n_particle,
//  RealIterator u, RealIterator v, RealIterator w,
//  RealIterator m,
//  size_t seed
//){
//}

// Future
//template<typename RealIterator, typename IntIterator>
//void relax(
//  size_t n_particle,
//  RealIterator u, RealIterator v, RealIterator w
//  RealIterator m,
//  IntIterator size_group,
//  size_t seed
//){
//}

} // END bphcuda
