#pragma once

#include <bphcuda/real.h>

#include <thrust/functional.h>
// #include <thrust/transform_reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

namespace bphcuda {

struct kinetic_e :public thrust::unary_function<Real3, Real> {
  __host__ __device__
  Real operator()(const Real3 &x){
    Real3 p = x * x;
    return p.get<0>() + p.get<1>() + p.get<2>();
  }
};

// input lists are [Real3]
__host__ __device__
template<typename Iter>
Real calc_kinetic_e(Iter cs_first, Iter cs_last){
  // Akira Hayakawa noted, 2010 10/14 14:27
  // transform_reduce seems having bug
  // return thrust::transform_reduce(cs_first, cs_last, kinetic_e(), 0.0F, thrust::plus<Real>());
  return thrust::reduce(
    thrust::make_transform_iterator(cs_first, kinetic_e()),
    thrust::make_transform_iterator(cs_last, kinetic_e()),
    0.0F,
    thrust::plus<Real>());
}

} // end of bphcuda
