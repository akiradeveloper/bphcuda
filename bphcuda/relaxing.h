#pragma once

#include <bphcuda/real.h>
#include <bphcuda/distribution.h>
#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

namespace bphcuda {

template<typename Velocity>
__host__ __device__
void relax(Velocity cs_F, Velocity cs_L, Int seed){
  Real old_kinetic = calc_kinetic_e(cs_F, cs_L);
  alloc_shell_rand(cs_F, cs_L, seed);  
  Real new_kinetic = calc_kinetic_e(cs_F, cs_L);
  Real ratio = old_kinetic / new_kinetic;
  thrust::transform(
    cs_F, cs_L,
    cs_F,
    multiplies(ratio));
}

} // end of bphcuda
