#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution/shell_distribution.h>

#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using namespace thrusting::op;
}

namespace bphcuda {

/*
  [real3] -> [real3]
  allocate new velocity to particle
  sustaining the momentum.
  This function assumes every particles are same in mass.
*/
template<typename RealIterator>
void alloc_new_c(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
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
  Akira Hayakawa 2010 11/13
  This implementation is wrong.
  This code will only run correctly in gravity center system.
  Which is understood equal ot local balancing system in this case.
  Assuming the input are in any system but,
  correcting them into gravity center system
  and then apply the procedure, recover the gravity center velocity.
  After all, this procedure will be more generic.
*/

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

namespace bphcuda {

/*
  Allocate new velocity all over the particle
  so that the momentum are preserved in each cell
*/
template<typename R1, typename R2, typename I1, typename RealTmp, typename IntTmp>
void alloc_new_c_all(
  size_t n_particle,
  R1 u, R1, v, R1 w, // input and output
  I1 idx, // input
  RealTmp tmp1, RealTmp tmp2, RealTmp tmp3, // output
  size_t n_cell, 
  IntTmp tmp4,
  RealTmp tmp5, RealTmp tmp6, RealTmp tmp7,
  size_t seed
){
  // alloc new velocity all over the particles
  alloc_new_c( 
    n_particle,
    u, v, w,
    seed);
  // calculate the average velocities in each cell
  thrust::reduce_by_key(
    idx,
    thrusting::advance(n_particle, idx),
    thrusting::make_zip_iterator(u, v, w),
    tmp4, // cell count
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7)); // sum velocity
  // averaging by cell count
  thrust::transform(
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7),
    thrusting::advance(n_cell, thrusting::make_zip_iterator(tmp5, tmp6, tmp7)),
    tmp4,
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7), // average velocity in cell
    thrusting::divides<real3, size_t>()); 
  // create n_particle length of average velocity in each cell
  thrust::gather(
    idx,
    thrusting::advance(n_particle, idx),
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7),
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3)); // the average velocity of n_particle length
  // modified velocity
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3),
    thrusting::make_zip_iterator(u, v, w),
    thrust::minus<real3>());
}

template<typename R1, typename R2>
void relax_all(
  size_t n_particle,
  R1 u, R1 v, R1 w,
  I1 idx,
  R3 tmp1, R3 tmp2, R3 tmp3,
  size_t n_cell,
  R3 tmp4, R3 tmp5, R3 tmp6,

} // END bphcuda
