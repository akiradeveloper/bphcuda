#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/tuple.h>

namespace {
  using namespace thrusting::op;
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real6;
}

namespace bphcuda {
/*
  (p, c) -> p
*/
struct move :public thrust::unary_function<real6, real3> {
  real _dt;
  move(real dt)
  :_dt(dt){}
  __device__ __host__
  real3 operator()(const real6 &in){
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + _dt * c;
  }
};

} // END bphcuda
