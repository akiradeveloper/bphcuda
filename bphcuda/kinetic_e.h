#pragma once

#include <bphcuda/real.h>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace bphcuda {

struct kinetic_e :public thrust::unary_function<Real3, Real> {
  __host__ __device__
  Real operator()(const Real3 &x){
    Real3 p = x * x;
    return p.get<0>() + p.get<1>() + p.get<2>();
  }
};

// input lists are [Real3]
template<typename Iter>
Real calc_kinetic_e(Iter cs_first, Iter cs_last){
  return thrust::transform_reduce(cs_first, cs_last, kinetic_e(), 0.0f, thrust::plus<Real>());
}

} // end of bphcuda
