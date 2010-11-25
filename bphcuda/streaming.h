#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/tuple.h>

/*
  moving functions are type
  (p, c, m) :: real7 -> (p, c) :: real6
*/

namespace {
  using namespace thrusting;
}

namespace bphcuda {

real6 make_real6(real3 a, real3 b){
  return real6(
    a.get<0>(), a.get<1>(), a.get<2>(),
    a.get<3>(), a.get<4>(), a.get<5>());
}

real7 make_real7(real6 a, real){
  return real7(
    a.get<0>(), a.get<1>(), a.get<2>(),
    a.get<3>(), a.get<4>(), a.get<5>(),
    a.get<6>());
}

/*
  1 step Runge Kutta.
*/
template<typename ForceGenerator>
struct _runge_kutta_1 :public thrust::unary_function<real7, real6> {
  ForceGenerator _f;
  real _dt;
  _move(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real7 &in){
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    real m = in.get<6>();
    real3 force = _f(in)
    real3 a = force / m;
    real3 new_p = p + _dt * c; // is this correct without 0.5 * m * a^2 ? 
    real3 new_c = c + _dt * a;
    return make_real6(new_p, new_c);
  }
};

template<typename ForceGenerator>
_runge_kutta_1<ForceGenerator> runge_kutta_1(ForceGenerator f, real dt){
  return _runge_kutta_1<ForceGenerator>(f, dt);
}

/*
  2 step Runge Kutta.
*/
template<typename ForceGenerator>
struct _runge_kutta_2 :public thrust::unary_function<real7, real6> {
  ForceGenerator _f;
  real _dt;
  _move(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real7 &in){
    /*
      h_ prefix means the value is at _dt/2 time passed.
    */
    // calc result by half time.
    real6 h_result = runge_kutta_1(_f, 0.5 * _dt)(in);
    // calc state of particle at the time half.
    real7 h_state = make_real7(h_result, m);
    real3 h_c = real3(h_state.get<3>(), h_state.get<4>(), h_state<5>());
    real3 h_force = _f(h_state);
    real3 h_a = h_force / m;
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>()); 
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    real3 new_p = p + _dt * h_c; // is this correct ? 
    real3 new_c = c + _dt * h_a;
    return make_real6(new_p, new_c);
  }
};

template<typename ForceGenerator>
_runge_kutta_2<ForceGenerator> runge_kutta_2(ForceGenerator f, real dt){
  return _runge_kutta_2<ForceGenerator>(f, dt);
}

} // END bphcuda
