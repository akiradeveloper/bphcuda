#pragma once

#include <bphcuda/real.h>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace bphcuda {

// c -> e
struct kinetic_e :public thrust::unary_function<Real3, Real> {
  __host__ __device__
  Real operator()(const Real3 &x){
    Real3 p = x * x;
    return p.get<0>() + p.get<1>() + p.get<2>();
  }
};

// [c] -> e
template<typename Velocity>
__host__ __device__
Real calc_kinetic_e(Velocity cs_F, Velocity cs_L){
  return thrust::transform_reduce(cs_F, cs_L, kinetic_e(), 0.0F, thrust::plus<Real>());
}

} // end of bphcuda
