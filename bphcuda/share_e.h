#pragma once

#include <thrusting/real.h>
#include <thrusting/functional.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/vectorspace.h>

#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using namespace thrusting;
}
  
namespace bphcuda {

/*
  (c, m, s) -> in_e 
*/
struct share_e_function :public thrust::unary_function<real5, real> {
  real operator()(const real5 &in) const {
    real4 x = real4(in.get<0>(), in.get<1>, in.get<2>(), in.get<3>()); // (c, m)
    real s = in.get<4>();
    return s * kinetic_e_calculator()(x) / real(3.0);
  }
};

} // END bphcuda
