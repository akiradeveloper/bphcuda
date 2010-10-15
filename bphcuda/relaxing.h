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

struct multiplies :public thrust::unary_function<Real3, Real3> {
  Real x;
  multiplies(Real x_)
  :x(x_){}
  __host__ __device__
  Real3 operator()(const Real3 &p){
    return x * p;
  }
};

template<typename Velocity>
__host__ __device__
void relax(Velocity cs_F, Velocity cs_L, Int seed){
  Real old_kinetic = calc_kinetic_e(cs_F, cs_L);
  alloc_shell_rand(cs_F, cs_L, seed);  
  Real new_kinetic = calc_kinetic_e(cs_F, cs_L);
  Real ratio = old_kinetic / new_kinetic;
  thrust::transform(
    cs_F, cs_L,
    thrust::make_constant_iterator(ratio3),
    cs_F,
    multiplies(ratio));
}

// Input : [(cs, ines)] : [tuple4], usually zip_iterator of float[] lists
template<typename Particle>
__host__ __device__
void share(Particle ps_F, Particle ps_F, Int s=0){
}

} // end of bphcuda
