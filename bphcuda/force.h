#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/functional.h>

/*
  ForceGenerator is represented by type
  (p, c, m) :: real7 -> force :: real3
*/

#include <iostream>

namespace {
  using namespace thrusting;
} 

namespace bphcuda {

struct no_force :public thrust::unary_function<real7, real3> {
  __host__ __device__
  real3 operator()(const real7 &in) const {
    return real3(0.0, 0.0, 0.0);
  }
};

namespace detail {
__device__ __host__
real calc_r3(real3 v3){
  real x = v3.get<0>();
  real y = v3.get<1>();
  real z = v3.get<2>();
  real r2 = x*x + y*y + z*z;
  return pow(r2, real(1.5));
}
} // END detail

struct gravitational_force :public thrust::unary_function<real7, real3> {
  real3 _P;
  real _M;
  real _G;
  gravitational_force(real3 P, real M, real G)
  :_P(P), _M(M), _G(G){}
  __host__ __device__
  real3 operator()(const real7 &in) const {
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>()); 
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    real m = in.get<6>();
    std::cout << m << std::endl;
    real3 vec_p = _P - p;
    real r3_vec_p = detail::calc_r3(vec_p);
    return _G * m * _M * vec_p / r3_vec_p;
  }
};

} // END bphcuda
