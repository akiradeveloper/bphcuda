#pragma once

#include <bphcuda/kinetic_e.h>

#include <thrusting/real.h>

#include <thrust/functional.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
/*
 * (c, m, in_e) -> total_e
 */
struct total_e_calculator :public thrust::unary_function<real5, real> {
  __host__ __device__
  real operator()(const real5 &in) const {
    real4 x(get<0>(in), get<1>(in), get<2>(in), get<3>(in));
    real kinetic_e = make_kinetic_e_calculator()(x);
    real in_e = get<4>(in);
    return kinetic_e + in_e;
  }
};
} // END detail

detail::total_e_calculator make_total_e_calculator(){
  return detail::total_e_calculator();
}

template<typename Real1, typename Real2>
real calc_total_e(
  size_t n_particle,
  Real1 u, Real1 v, Real1 w,
  Real2 m, Real1 in_e
){
  return thrust::transform_reduce(
    thrusting::make_zip_iterator(u, v, w, m, in_e),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m, in_e)),
    bphcuda::make_total_e_calculator(),
    real(0),
    thrust::plus<real>());   
}

} // END bphcuda
