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
  Allocate new velocity all over the particle
  so that the momentum are preserved in each cell
*/
template<typename R1, typename R2, typename I1, typename RealTmp, typename IntTmp>
void alloc_new_c_all(
  size_t n_particle,
  R1 u, R1, v, R1 w, // input and output
  I1 idx, // input
  /*
    can be removed
  */
  /*
    this can be removed by permutation_iterator
  */
  RealTmp tmp1, RealTmp tmp2, RealTmp tmp3, // output
  size_t n_cell, 
  RealTmp tmp5, RealTmp tmp6, RealTmp tmp7,
  IntTmp tmp4,
  size_t seed
){
  // alloc shell velocity all over the particles
  alloc_shell_rand(
    n_particle,
    u, v, w,
    seed); 

  /*
    calculate the average velocities in each cell
  */
  thrust::reduce_by_key(
    idx,
    thrusting::advance(n_particle, idx),
    thrusting::make_zip_iterator(u, v, w),
    tmp4, // cell count
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7)); // sum velocity

  /* 
    averaging by cell count
  */
  thrust::transform(
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7),
    thrusting::advance(n_cell, thrusting::make_zip_iterator(tmp5, tmp6, tmp7)),
    tmp4,
    thrusting::make_zip_iterator(tmp5, tmp6, tmp7), // average velocity in cell
    thrusting::divides<real3, size_t>()); 

  /*
     modified the velocity by the average velocity of each cell
  */ 
  
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
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // this line will be reimplemented by permutation_iterator
    thrusting::make_zip_iterator(u, v, w),
    thrust::minus<real3>());
}

struct pow_e :public thrust::unary_function<real3, real> {
  real operator()(const real3 &in) const {
    real x = in.get<0>();
    real y = in.get<1>();
    real z = in.get<2>();
    return x*x + y*y + z*z;
  }
};

struct sqrt :public thrust::unary_function<real, real> {
  real operator()(real x) const {
    return sqrt(x);
  }
};

template<typename R1, typename R2>
void relax_particle_parallel (
  size_t n_particle,
  R1 u, R1 v, R1 w,
  I1 idx,
  // This can be removed by using permutation_iterator
  RealTmp tmp1, RealTmp tmp2, RealTmp tmp3,
  size_t n_cell,
  RealTmp tmp4, 
  RealTmp tmp5, RealTmp tmp6, RealTmp tmp7,
  IntTmp tmp8,
  size_t seed
){
  // calc_total_e ahead of allocating new velocity
  thrust::reduce_by_key(
    idx,
    thrusting::advance(n_particle, idx),
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w),
      pow_e()),
    tmp8, // cell count
    tmp4); // old energy

  // allocate new velocity preserving the momentum in each cell
  alloc_new_c_all(
    n_particle,
    u, v, w,
    idx,
    tmp1, tmp2, tmp3, // can be removed
    n_cell,
    tmp5, tmp6, tmp7,
    tmp8, // reuse
    seed);

  // again, calc_e_total_e
  thrust::reduce_by_key(
    idx,
    thrusting::advance(n_particle, idx),
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w),
      pow_e()),
    tmp8, // cell count, again
    tmp5); // new energy. reuse

  // the ratio_e = old / new
  thrust::transform(
    tmp4,
    thrusting::advance(n_cell, tmp4),
    tmp5,
    tmp4,
    thrust::divides<real>());

  // sqrt it
  thrust::transform(
    tmp4,
    thrusting::advance(n_cell, tmp4),
    tmp4,
    sqrt());  

  // this procedure can be reduce by permutation_iterator
  // the ratio_e all over the particle 
  thrust::gather(
    idx,
    thrusting::advance(n_particle, idx),
    tmp4,
    tmp1); // reusing tmp1, sqrt of ratio_e

  // modified the velocity by ratio_e
  thrust::transform(
    tmp1, // TODO use permutation_iterator
    thrusting::advance(n_particle, tmp1),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::multiplies<real, real3>());   
}

} // END bphcuda
