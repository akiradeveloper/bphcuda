#pragma once

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include <thrusting/iterator.h>
#include <thrusting/real.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/functional.h>

#include <bphcuda/kinetic_e.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

/*
  (c, m, s) -> in_e
*/
struct in_e_allocator :public thrust::unary_function<real5, real> {
  __host__ __device__
  real operator()(const real5 &in) const {
    real3 c = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real m = in.get<3>();
    real s = in.get<4>();
    real ratio = s / real(3.0);
    return ratio * calc_kinetic_e(c, m);
  }
};

template<typename Real1, typename Real2, typename Real3>
void alloc_in_e(
  size_t n_particle, 
  Real1 u, Real1 v, Real1 w,
  Real2 m,
  Real1 in_e, // output
  Real3 s
){
  thrust::transform(
    thrusting::make_zip_iterator(u, v, w, m, s),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m, s)),
    in_e,
    in_e_allocator());  
}

} // END bphcuda
