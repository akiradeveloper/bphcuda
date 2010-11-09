#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include <thrusting/dtype/real.h>

namespace {
  using thrusting::real;
}

namespace bphcuda {

/*
  Initialization scheme.
  Inner energies are generated by sharing the kinetic energy
  that is allocated by distribution function initially. 
*/
template<typename RealIteartor>
void alloc_in_e(
  size_t n_p, 
  RealIteartor u, RealIteartor v, RealIteartor w,
  RealIteartor in_e, // output
  RealIterator s, 
  RealIteartor m,
){
}

// deprecated
template<typename Velocity, typename InE>
void alloc_ine(Velocity cs_F, Velocity cs_L, InE ines_F, Int s){
  Real ratio = s / 3.0F;
  transform(
    thrust::make_transform_iterator(cs_F, kinetic_e()),
    thrust::make_transform_iterator(cs_L, kinetic_e()),
    ines_F,
    thrust::multiplies<Real>());
}

} 
