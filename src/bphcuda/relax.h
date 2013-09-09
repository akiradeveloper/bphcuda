#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/algorithm/reduce_by_bucket.h>
#include <thrusting/list.h>
#include <thrusting/algorithm/logical.h>
#include <thrusting/assert.h>
#include <thrusting/pp.h>

#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution/shell_distribution.h>
#include <bphcuda/total_e.h>

#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>

/*
  If cnt = 0 or 1 do nothing.
  cnt = 1 then no relax will happen.
*/

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {

struct is_nan :public thrust::unary_function<real, bool> {
  __host__ __device__
  bool operator()(real x) const {
    return isnan(x);
  }
};

/*
  Allocate new velocity all over the particle
  so that the momentum are conserved in each cell
*/
template<typename Real, typename Int>
void alloc_new_c_all(
  size_t n_particle,
  Real u, Real v, Real w, // input and output
  Int idx, // input
  size_t n_cell, 
  Real tmp1, Real tmp2, Real tmp3, Real tmp6, Real tmp7, Real tmp8,
  Int tmp4, Int tmp5,
  size_t seed
){
  thrusting::bucket_indexing(
    n_particle, 
    idx,
    n_cell,
    tmp4,
    tmp5); // cnt
  
  /*
    if cnt > 1 then alloc shell rand
  */
  alloc_shell_rand_if(
    n_particle,
    u, v, w,
    thrust::make_permutation_iterator(tmp5, idx), // stencil
    thrusting::bind2nd(thrust::greater<size_t>(), 1), // if cnt > 1 
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
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // velocity sum
    thrusting::make_zip_iterator(tmp6, tmp7, tmp8), // tmp
    zero_veloc); 

  /* 
    averaging by cell count
  */
  thrust::transform_if(
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // velocity sum
    thrusting::advance(n_cell, thrusting::make_zip_iterator(tmp1, tmp2, tmp3)),
    tmp5, // cnt
    tmp5, // stencil
    thrusting::make_zip_iterator(tmp1, tmp2, tmp3), // output, average velocity in cell
    thrusting::divides<real3, size_t>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1)); 

  /*
    minus average velocity
    if cnt > 1
  */
  thrust::transform_if(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrust::make_permutation_iterator( // input2
      thrusting::make_zip_iterator(tmp1, tmp2, tmp3),
      idx),
    thrust::make_permutation_iterator( // stencil
      tmp5, // cnt
      idx),
    thrusting::make_zip_iterator(u, v, w), // output
    thrust::minus<real3>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1)); // if not cnt = 1
}

struct RELAX_SQRT :public thrust::unary_function<real, real> {
  __host__ __device__
  real operator()(real x) const {
    return sqrt(x);
  }
};
} // END detail

template<typename Real, typename Int>
void relax (
  size_t n_particle,
  Real u, Real v, Real w,
  real m,
  Real in_e,
  real s,
  Int idx,
  size_t n_cell,
  Real tmp1, Real tmp2, Real tmp3, Real tmp4, Real tmp7, 
  Real tmp8, Real tmp9,
  Int tmp5, Int tmp6,
  size_t seed
){
  thrust::constant_iterator<real> m_it(m);

  real zero_e(0.0);
  /*
    calc E_kin before allocating new velocity
  */
  thrusting::reduce_by_bucket(
    n_particle,
    idx,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w, m_it, in_e),
      make_total_e_calculator()),
    n_cell,
    tmp5, // tmp
    tmp6, // tmp
    tmp1, // total_e by cell
    tmp2, // tmp
    zero_e); // if cell is empty, the total_e is 0

  detail::alloc_new_c_all(
    n_particle,
    u, v, w, // new c allocated. momentum are 0
    idx,
    n_cell,
    tmp2, tmp3, tmp4, tmp7, tmp8, tmp9, // tmp
    tmp5, tmp6, // tmp
    seed);

  /*
    calc E_kin after allocating new velocity by cell
  */
  thrusting::reduce_by_bucket(
    n_particle,
    idx,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(u, v, w, m_it),
      make_kinetic_e_calculator()),
    n_cell,
    tmp5, // tmp
    tmp6, // cnt
    tmp2, // kinetic_e by cell
    tmp3, // tmp
    zero_e);  // if cell is empty, the total_kinetic_e is 0

  /*
    creating scheduled kinetic_e
  */
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp3, // output, scheculed kinetic_e by cell
    thrusting::bind1st(thrust::multiplies<real>(), real(3) / (real(3) + s)));

  thrust::transform_if(
    tmp3, // scheduled kinetic_e by cell
    thrusting::advance(n_cell, tmp3),
    tmp2, // kinetic_e by cell
    tmp6, // stencil, cnt
    tmp3, // output, ratio_kinetic_e
    thrust::divides<real>(), 
    thrusting::bind2nd(thrust::greater<size_t>(), 1)); // if cnt > 1
  
  /*
    sqrt it
  */
  thrust::transform_if(
    tmp3,
    thrusting::advance(n_cell, tmp3),
    tmp6, // stencil, cnt
    tmp3, // ratio_c
    detail::RELAX_SQRT(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));  

  /*
    multiplies ratio_c 
  */
  thrust::transform_if(
    thrust::make_permutation_iterator(tmp3, idx), // input1, ratio_c 
    thrusting::advance(
      n_particle, 
      thrust::make_permutation_iterator(tmp3, idx)),
    thrusting::make_zip_iterator(u, v, w), // input2
    thrust::make_permutation_iterator(tmp6, idx), // stencil
    thrusting::make_zip_iterator(u, v, w), // output
    thrusting::multiplies<real, real3>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));   

  /*
    creating new in_e by cell
  */
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp4, // output, total_in_e by cell
    thrusting::bind1st(thrust::multiplies<real>(), s / (real(3) + s)));
  
  thrust::transform_if(
    tmp4,
    thrusting::advance(n_cell, tmp4),
    tmp6, // denom, cnt
    tmp6, // stencil
    tmp4, // output, particle's in_e by cell. in_e by particle is unique.
    thrusting::divides<real, size_t>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));

  thrusting::transform_if(
    n_particle, 
    thrust::make_permutation_iterator(tmp4, idx), // input, new in_e by particle
    thrust::make_permutation_iterator(tmp6, idx), // stencil, cnt
    in_e,
    thrust::identity<real>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));
}

} // END bphcuda
