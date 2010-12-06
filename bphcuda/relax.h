#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/algorithm/reduce_by_bucket.h>
#include <thrusting/list.h>

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
  Real tmp1, Real tmp2, Real tmp3,
  Int tmp4,
  Int tmp5,
  size_t seed
){
  // std::cout << "begin alloc_new_c_all" << std::endl;
  thrusting::bucket_indexing(
    n_particle, 
    idx,
    n_cell,
    tmp4,
    tmp5); // cnt
  
  // std::cout << make_list(n_cell, tmp5) << std::endl;
    
  /*
    if cnt > 1 then alloc shell rand
  */
  alloc_shell_rand_if(
    n_particle,
    u, v, w,
    thrust::make_permutation_iterator(tmp5, idx), // stencil
    thrusting::bind2nd(thrust::greater<size_t>(), 1), // if cnt > 1 
    seed); 
  
  // std::cout << make_list(n_particle, u) << std::endl; 

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
    zero_veloc); 

  // std::cout << make_list(n_cell, tmp1) << std::endl;
  // std::cout << make_list(n_cell, tmp5) << std::endl;

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

  // std::cout << make_list(n_cell, tmp1) << std::endl;

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

  // std::cout << make_list(n_particle, u) << std::endl;
  // std::cout << "end alloc_new_c_all" << std::endl;
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
  Real tmp1,
  Real tmp2, Real tmp3, Real tmp4,
  Int tmp5,
  Int tmp6,
  size_t seed
){
  std::cout << "begin relax" << std::endl;
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
    tmp5, 
    tmp6, // cnt
    tmp1, // total_e by cell
    zero_e); // if cell is empty, the total_e is 0

  // OK
  std::cout << make_list(n_cell, tmp6) << std::endl; 
  std::cout << make_list(n_cell, tmp1) << std::endl;

  detail::alloc_new_c_all(
    n_particle,
    u, v, w, // new c allocated. momentum are 0
    idx,
    n_cell,
    tmp2, tmp3, tmp4,
    tmp5, 
    tmp6,
    seed);

  // OK
  std::cout << "u: " << make_list(n_particle, u) << std::endl;
  std::cout << "v: " << make_list(n_particle, v) << std::endl;
  std::cout << "w: " << make_list(n_particle, w) << std::endl;
  
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
    tmp5, 
    tmp6, // cnt
    tmp2, // tmp kinetic_e by cell
    zero_e);  // if cell is empty, the total_kinetic_e is 0

//  std::cout << make_list(n_cell, tmp6) << std::endl;
  std::cout << "tmp_e_kin: " << make_list(n_cell, tmp2) << std::endl;

  /*
    creating scheduled kinetic_e
  */
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp3, // output, scheculed kinetic_e by cell
    thrusting::bind1st(thrust::multiplies<real>(), real(3) / (real(3) + s)));
  
  std::cout << "sheduled e_kin: " << make_list(n_cell, tmp3) << std::endl;

  thrust::transform_if(
    tmp3, // scheduled kinetic_e by cell
    thrusting::advance(n_cell, tmp3),
    tmp2, // tmp kinetic_e by cell
    tmp6, // stencil, cnt
    tmp3, // output, ratio_kinetic_e
    thrust::divides<real>(), 
    thrusting::bind2nd(thrust::greater<size_t>(), 1)); // if cnt > 1

  std::cout << "after divides: " << make_list(n_cell, tmp3) << std::endl;
  
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

  std::cout << "after sqrt: " << make_list(n_cell, tmp3) << std::endl;

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

  std::cout << "u: " << make_list(n_particle, u) << std::endl;
  std::cout << "v: " << make_list(n_particle, v) << std::endl;
  std::cout << "w: " << make_list(n_particle, w) << std::endl;

  /*
    creating new in_e by cell
  */
  thrust::transform(
    tmp1,
    thrusting::advance(n_cell, tmp1),
    tmp4, // output, total_in_e by cell
    thrusting::bind1st(thrust::multiplies<real>(), s / (real(3) + s)));
  
  std::cout << "new in_e: " << make_list(n_cell, tmp4) << std::endl;

  thrust::transform_if(
    tmp4,
    thrusting::advance(n_cell, tmp4),
    tmp6, // denom, cnt
    tmp6, // stencil
    tmp4, // output, particle's in_e by cell. in_e by particle is unique.
    thrusting::divides<real, size_t>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));

  std::cout << "new in_e per particle: " << make_list(n_cell, tmp4) << std::endl;

  thrusting::transform_if(
    n_particle, 
    thrust::make_permutation_iterator(tmp4, idx), // input, new in_e by particle
    thrust::make_permutation_iterator(tmp6, idx), // stencil, cnt
    in_e,
    thrust::identity<real>(),
    thrusting::bind2nd(thrust::greater<size_t>(), 1));
  
  std::cout << "end relax" << std::endl;
}

} // END bphcuda
