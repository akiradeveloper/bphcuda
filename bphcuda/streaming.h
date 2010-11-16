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
  (p, c) -> (p, c)
  1 step Runge Kutta.
*/
template<typename ForceGenerator>
struct _runge_kutta_1 :public thrust::unary_function<real6, real6> {
  ForceGenerator _f;
  real _dt;
  _move(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real6 &in){
    // Wrong Impl
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + _dt * c;
  }
};

template<typename ForceGenerator>
_runge_kutta_1<ForceGenerator> runge_kutta_1(ForceGenerator f, real dt){
  return _runge_kutta_1<ForceGenerator>(f, dt);
}

/*
  (p, c) -> (p, c) 
  2 step Runge Kutta.
*/
template<typename ForceGenerator>
struct _runge_kutta_2 :public thrust::unary_function<real6, real6> {
  ForceGenerator _f;
  real _dt;
  _move(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real6 &in){
    // Wrong
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + _dt * c;
  }
};

template<typename ForceGenerator>
_runge_kutta_2<ForceGenerator> runge_kutta_2(ForceGenerator f, real dt){
  return _runge_kutta_2<ForceGenerator>(f, dt);
}

} // END bphcuda
