#pragma once

#include <bphcuda/real.h>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace bphcuda {

// (c, m) -> e
struct kinetic_e :public thrust::unary_function<Real4, Real> {
  __host__ __device__
  Real operator()(const Real4 &x){
    Real3 v = mk_real3(x.get<0>(), x.get<1>(), x.get<2>());
    Real3 p = v * v;
    Real m = x.get<3>();
    return 0.5 * m * (p.get<0>() + p.get<1>() + p.get<2>());
  }
};

// [(c,m)] -> e
template<typename Particle>
Real calc_kinetic_e(Particle F, Particle L){
  return thrust::transform_reduce(F, L, kinetic_e(), 0.0F, thrust::plus<Real>());
}

} // end of bphcuda
