#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>

/*
  ForceGenerator is represented by type
  (p, c, m) :: real7 -> force :: real3
*/

namespace {
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real6;
  using thrusting::real7;
  using namespace thrusting::op;
} 

struct no_force :public thrust::unary_function<real7, real3> {
  real3 operator()(const &in) const {
    return real3(0.0, 0.0, 0.0);
  }
};

real calc_r3(real3 v3){
  real x = v3.get<1>();
  real y = v3.get<2>();
  real z = v3.get<3>();
  real r2 = x*x + y*y + z*z;
  return pow(r2, 1.5);
}

struct gravitational_force :public thrust::unary_function<real7, real3> {
  real3 _p;
  real _M;
  gravitational_force(real3 p, real M)
  :_p(p), _M(M){}
  real3 operator()(const &in) const {
    real G = GRAVITATIONAL_CONST();
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>()); 
    real3 c = real3(in.get<1>(), in.get<2>(), in.get<3>());
    real m = in.get<6>();
    real3 vec_p = _p - p;
    real r3_vec_p = calc_r3(vec_p);
    return G * m * _M * vec_p / r3_vec_p;
  }
};
