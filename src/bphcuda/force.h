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
  using namespace thrust;
  using namespace thrusting;
} 

namespace bphcuda {

namespace  detail {
struct no_force :public thrust::unary_function<real7, real3> {
  __host__ __device__
  real3 operator()(const real7 &in) const {
    return real3(0.0, 0.0, 0.0);
  }
};
} // END detail

detail::no_force make_no_force_generator(){
  return detail::no_force();
}

namespace detail {
__device__ __host__
real calc_r3(real3 v3){
  real x = get<0>(v3);
  real y = get<1>(v3);
  real z = get<2>(v3);
  real r2 = x*x + y*y + z*z;
  return pow(r2, real(1.5));
}

class gravitational_force :public thrust::unary_function<real7, real3> {
  real3 _P;
  real _M;
  real _G;
public:
  gravitational_force(real3 P, real M, real G)
  :_P(P), _M(M), _G(G){}
  __host__ __device__
  real3 operator()(const real7 &in) const {
    real3 p(get<0>(in), get<1>(in), get<2>(in)); 
    real3 c(get<3>(in), get<4>(in), get<5>(in));
    real m = get<6>(in);
    real3 vec_p = _P - p;
    real r3_vec_p = detail::calc_r3(vec_p);
    return _G * m * _M * vec_p / r3_vec_p;
  }
};
} // END detail

detail::gravitational_force make_gravitational_force_generator(real3 P, real M, real G){
  return detail::gravitational_force(P, M, G);
}

} // END bphcuda
