#pragma once

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <thrust/transform_reduce.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
__host__ __device__
real calc_kinetic_e(const real3 &c, real m){
  return real(0.5) * m * (
    c.get<0>() * c.get<0>() +
    c.get<1>() * c.get<1>() +
    c.get<2>() * c.get<2>());
}

/*
  (c, m) -> kinetic_e
*/
struct kinetic_e_calculator :public thrust::unary_function<real4, real> {
  __host__ __device__
  real operator()(const real4 &in) const {
    real3 c(in.get<0>(), in.get<1>(), in.get<2>());
    real m(in.get<3>());
    return calc_kinetic_e(c, m);
  }
}; 
} // END detail

detail::kinetic_e_calculator make_kinetic_e_calculator(){
  return detail::kinetic_e_calculator();
}

/*
  [(c, m)] -> [kinetic_e]
*/
template<typename Real1, typename Real2>
real calc_kinetic_e(
  size_t n_particle, 
  Real1 u, Real1 v, Real1 w, 
  Real2 m
){
  return thrust::transform_reduce(
    thrusting::make_zip_iterator(u, v, w, m),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m)),
    make_kinetic_e_calculator(),
    real(0.0),
    thrust::plus<real>());
}

} // END bphcuda
