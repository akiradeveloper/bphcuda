#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>

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
  RealIteartor u, RealIteartor v, RealIteartor w
){
}

template<typename RealIterator>
void alloc_new_c(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  RealIterator m
){
}

} // END namespace

namespace bphcuda {

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
  RealIterator u, RealIterator v, RealIterator w
){
}

// Future
template<typename RealIterator>
void relax(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  RealIterator m
){
}

// Future
template<typename RealIterator, typename IntIterator>
void relax(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w
  RealIterator m,
  IntIterator n_group
){
}

// deprecated
// [Real3] -> [Real3]
template<typename Velocity>
void _alloc_new_c(Velocity cs_F, Velocity cs_L, Int seed){
  int len = cs_L - cs_F;
  *(cs_L-1) = mk_real3(0.0,0.0,0.0);
  int h_len = len / 2;
  alloc_shell_rand(cs_F, cs_F+h_len, seed);
  thrust::copy(cs_F, cs_F+h_len, cs_F+h_len); 
  thrust::transform(cs_F, cs_F+h_len, cs_F, multiplies(-1));
}
  
// deprecated
// [Real3] -> [Real3]
template<typename Velocity>
void relax(Velocity cs_F, Velocity cs_L, Int seed){
  int len = cs_L - cs_F;
  if(len < 2){ return; }
  Real old_kinetic = calc_kinetic_e(cs_F, cs_L);
  _alloc_new_c(cs_F, cs_L, seed);  
  Real new_kinetic = calc_kinetic_e(cs_F, cs_L);
  Real ratio = sqrt(old_kinetic / new_kinetic);
  thrust::transform(
    cs_F, cs_L,
    cs_F,
    multiplies(ratio));
}

} // end of bphcuda
