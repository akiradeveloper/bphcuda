#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/tuple.h>

namespace {
  using namespace thrusting::op;
  using thrusting::real;
  using thrusting::real3;
}

namespace bphcuda {

// (p, c) -> p
struct move :public thrust::unary_function<real6, real3> {
  real _dt;
  move(real dt)
  :_dt(dt){}

  __device__ __host__
  real3 operator()(const real6 &in){
    real3 p = thrusting::make_real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = thrusting::make_real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + _dt * c;
  }
};

// deprecated
// Have to be Real6 -> Real6?
/*
Input [(xs, cs)]
Output [(xs)]
*/
struct move :public thrust::unary_function<Real6, Real3>{
  Real dt;
  move(Real dt_)
  :dt(dt_){}
  __device__ __host__
  Real3 operator()(const Real6 &in){
    Real3 p = mk_real3(in.get<0>(), in.get<1>(), in.get<2>());
    Real3 c = mk_real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + c * dt;
  }
};

} // END bphcuda
