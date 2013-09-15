#pragma once

#include <thrust/transform_reduce.h>

#include <thrusting/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/vectorspace.h>
#include <thrusting/tuple.h>
#include <thrusting/iterator.h>

namespace {
  using namespace thrust;
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
/*
 * (c, m) -> momentum
 */
struct momentum_calculator :public thrust::unary_function<real4, real3>{
  __host__ __device__
  real3 operator()(const real4 &in) const {
    real3 c = real3(get<0>(in), get<1>(in), get<2>(in));
    real m = get<3>(in);
    return m * c;
  }
};
} // END detail

/*
 * [(c, m)] -> [momentum]
 */
template<typename Real1, typename Real2>
real3 calc_momentum(
  size_t n_particle,
  Real1 u, Real1 v, Real1 w, 
  Real2 m
){
  return thrust::transform_reduce(
    thrusting::make_zip_iterator(u, v, w, m),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m)),
    detail::momentum_calculator(),
    real3(0.0, 0.0, 0.0),
    tuple3plus<real3>());
}

} // END bphcuda
