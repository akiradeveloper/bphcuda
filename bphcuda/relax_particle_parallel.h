#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/algorithm/reduce_by_bucket.h>

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
template<typename Real, typename Int>
void alloc_new_c_all(
  size_t n_particle,
  Real u, Real v, Real w, // input and output
  Int idx, // input
  size_t n_cell, 
  Real tmp1, Real tmp2, Real tmp3,
  Int tmp4,
  Int tmp5,
  size_t seed
){
  // alloc shell velocity all over the particles
  alloc_shell_rand(
    n_particle,
    u, v, w,
    seed); 

  real3 zero_veloc(0.0,0.0,0.0);
  /*
    calculate the average velocities in each cell
  */
  thrusting::reduce_by_bucket(
    n_particle,
    idx,
    thrusting::make_zip_iterator(u, v, w),
    n_cell,
    tmp4, 
    tmp5, // cnt
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3)
    zero_veloc); // velocity sum

  /* 
    averaging by cell count
    if stencil not 0 then divides
  */
  thrust::transform_if(
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // velocity sum
    thrusting::advance(n_cell, thrusting::make_zip_iterator(tmp1, tmp2, tmp3)),
    tmp5, // cnt
    tmp5, // stencil
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // average velocity in cell
    thrusting::divides<real3, size_t>(),
    thrusting::bind2nd(thrust::not_equal_to<size_t>(), 0)); 

  /*
    minus average velocity
  */
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_permutation_iterator(
      thrusting::make_zip_iterator(tmp1, tmp2, tmp3),
      idx),
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

template<typename Real, typename Int>
void relax_particle_parallel (
  size_t n_particle,
  Real u, Real v, Real w,
  Real in_e,
  real s,
  Int idx,
  size_t n_cell,
  Real tmp1
  Real tmp2, Real tmp3, Real tmp4,
  Int tmp5,
  Int tmp6,
  size_t seed
){
  real zero_e(0.0);
  /*
    calc_total_e ahead of allocating new velocity
  */
  thrust::reduce_by_bucket(
    n_particle,
    idx,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w),
      pow_e()),
    n_cell,
    tmp5, // prefix
    tmp6, // cnt
    tmp1, // old energy by cell
    zero_e);

  // allocate new velocity preserving the momentum in each cell
  alloc_new_c_all(
    n_particle,
    u, v, w,
    idx,
    n_cell,
    tmp2, tmp3, tmp4,
    tmp5, 
    tmp6,
    seed);

  /*
    again, calc_e_total_e
  */
  thrust::reduce_by_bucket(
    n_particle,
    idx,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w),
      pow_e()),
    n_cell,
    tmp5, 
    tmp6, // cnt
    tmp2, // new energy by cell
    zero_e); 

  // or the stencil can be either tmp1 or tmp2 for not 0 to divides
  /*
    the ratio_e = old / new
    if cnt not 0 then divides
  */
  thrust::transform_if(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp2,
    tmp6, // stencil, cnt
    tmp1, // ratio_e
    thrust::divides<real>(),
    thrusting::bind2nd(thrust::not_equal_to<size_t>(), 0));
  
  // this can be evaluate lazily
  /*
    sqrt it
  */
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp1,
    sqrt());  

  /*
    multiplies ratio_veloc
  */
  thrust::transform(
    thrust::make_permutation_iterator(tmp1, idx),  
    thrusting::advance(
      n_particle, 
      thrust::make_permutation_iterator(tmp1, idx)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::multiplies<real, real3>());   

  /*
    share
  */
  thrust::constant_iterator<real> m_it(1);
  thrust::constant_iterator<real> s_it(s);
   
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w, m_it, s_it),
    thrusting::advance(
      n_particle,
      thrusting::make_zip_iterator(u, v, w, m_it, s_it)),
    in_e,
    share_e_function());
}

} // END bphcuda
