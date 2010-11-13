#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using namespace thrusting::op;
}
  
namespace bphcuda {

/*
  Reallcate total energy of one particle
  so that total energy to be shared in 3:s between kinetic_e and inner thermal energy.
*/
template<typename RealIterator1, typename RealIterator2, typename RealIterator3>
void share_e(
  size_t n_particle, 
  RealIterator1 u, RealIterator1 v, RealIterator1 w, // input and output
  RealIterator2 m, 
  RealIterator3 in_e, // input and output 
  real s 
){
  real old_kinetic_e = calc_kinetic_e(n_particle, u, v, w, m);
  real old_in_e = thrust::reduce(
    in_e,
    thrusting::advance(n_particle, in_e),
    real(0.0));
  real total_e = old_kinetic_e + old_in_e;
  real new_kinetic_e = (real(3.0) / (real(3.0) + s)) * total_e;
  real new_in_e = (s / (real(3.0) + s)) * total_e;
  
  real ratio_c = sqrt(new_kinetic_e / old_kinetic_e);
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), ratio_c));

  real ratio_in_e = new_in_e / old_in_e;
  thrust::transform(
    in_e,
    thrusting::advance(n_particle, in_e),
    in_e,
    thrusting::bind1st(thrusting::multiplies<real, real>(), ratio_in_e)); 
}

} // END bphcuda
