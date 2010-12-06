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
  (c, m, in_e) -> total_e
*/
struct total_e_calculator :public thrust::unary_function<real5, real> {
  __host__ __device__
  real operator()(const real5 &in) const {
    real4 x(in.get<0>(), in.get<1>(), in.get<2>(), in.get<3>());
    real kinetic_e = make_kinetic_e_calculator()(x);
    real in_e = in.get<4>();
    return kinetic_e + in_e;
  }
};
} // END detail

detail::total_e_calculator make_total_e_calculator(){
  return detail::total_e_calculator();
}

} // END bphcuda
