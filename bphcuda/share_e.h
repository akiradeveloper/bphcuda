#pragma once

#include <thrusting/dtype/real.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
}
  
namespace bphcuda {

/*
  Reallcate total energy of one particle
  so that total energy to be shared in 3:s between kinetic_e and inner thermal energy.
*/
template<typename RealIterator>
void share_e(
  size_t n_particle, 
  RealIterator u, RealIterator v, RealIterator w, // input and output
  RealIterator m, 
  RealIterator in_e, // input and output 
  real s 
){
  real old_kinetic_e = calc_kinetic_e(n_particle, u, v, w, m);
  real old_in_e = thrust::reduce(
    in_e,
    thrusting::advance(n_particle, in_e),
    0.0);
  real total_e = old_kinetic_e + old_in_e;
  real new_kinetic_e = (3.0 / (3.0 + s)) * total_e;
  real new_in_e = (s / (3.0 + s)) * total_e;
  
  real ratio_c = sqrt(new_kinetic_e / old_kinetic_e);
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w),
    thrusting::advance(n_particle, thrusing::make_zip_iterator(u, v, w)),
    thrusting::make_zip_iterator(u, v, w),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), ratio_c));

  real ratio_in_e = new_in_e / old_in_e;
  thrust::transform(
    in_e,
    thrusting::advance(n_particle, in_e),
    in_e,
    thrusting::bind1st(thrusting::multiplies<real, real>(), ratio_in_e)); 
}
