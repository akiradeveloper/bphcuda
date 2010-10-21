#pragma once

#include <bphcuda/real.h>
#include <bphcuda/kinetic_e.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

namespace bphcuda {

template<typename Velocity, typename InE>
void alloc_ine(Velocity cs_F, Velocity cs_L, InE ines_F, Int s){
  Real ratio = s / 3.0F;
  transform(
    thrust::make_transform_iterator(cs_F, kinetic_e()),
    thrust::make_transform_iterator(cs_L, kinetic_e()),
    ines_F,
    thrust::multiplies<Real>());
}

} // end of bphcuda
