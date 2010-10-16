#pragma once

#include <bphcuda/real.h>
#include <bphcuda/distribution.h>
#include <bphcuda/kinetic_e.h>
#include <bphcuda/distribution.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>

namespace bphcuda {

// [Real3] -> [Real3]
template<typename Velocity>
void _alloc_new_c(Velocity cs_F, Velocity cs_L, Int seed){
  int len = cs_L - cs_F;
  *(cs_L-1) = mk_real3(0.0,0.0,0.0);
  int h_len = len / 2;
  alloc_shell_rand(cs_F, cs_F+h_len, seed);
  thrust::copy(cs_F, cs_F+h_len, cs_F+h_len); 
  thrust::transform(cs_F, cs_F+h_len, cs_F, multiplies(-1));
}
  
// [Real3] -> [Real3]
template<typename Velocity>
void relax(Velocity cs_F, Velocity cs_L, Int seed){
  int len = cs_L - cs_F;
  if(len < 2){ return; }
  Real old_kinetic = calc_kinetic_e(cs_F, cs_L);
  _alloc_new_c(cs_F, cs_L, seed);  
  Real new_kinetic = calc_kinetic_e(cs_F, cs_L);
  Real ratio = sqrt(old_kinetic / new_kinetic);
  thrust::transform(
    cs_F, cs_L,
    cs_F,
    multiplies(ratio));
}

} // end of bphcuda
