#pragma once

#include <thrust/transform_reduce.h>

#include <thrusting/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/tuple.h>
#include <thrusting/iterator.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

/*
  (c, m) -> momentum
*/
struct momentum_calculator :public thrust::unary_function<real4, real3>{
  __host__ __device__
  real3 operator()(const real4 &in) const {
    real3 c = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real m = in.get<3>();
    return m * c;
  }
};

/*
  [(c, m)] -> [momentum]
*/
template<typename Real, typename Real2>
real3 calc_momentum(
  size_t n_particle,
  Real u, Real v, Real w, 
  Real2 m
){
  return thrust::transform_reduce(
    thrusting::make_zip_iterator(u, v, w, m),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m)),
    momentum_calculator(),
    real3(0.0, 0.0, 0.0),
    thrust::plus<real3>());
}

} // END bphcuda
