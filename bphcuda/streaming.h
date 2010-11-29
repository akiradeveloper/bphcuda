#pragma once

#include <thrusting/real.h>
#include <thrusting/vectorspace.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>

/*
  moving functions are type
  (p, c, m) :: real7 -> (p, c) :: real6
*/

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
__host__ __device__
real6 make_real6(const real3 &a, const real3 &b){
  return real6(
    a.get<0>(), a.get<1>(), a.get<2>(),
    b.get<0>(), b.get<1>(), b.get<2>());
}

__host__ __device__
real7 make_real7(const real6 &a, const real &b){
  return real7(
    a.get<0>(), a.get<1>(), a.get<2>(),
    a.get<3>(), a.get<4>(), a.get<5>(),
    b);
}
} // END detail

namespace detail {
/*
  1 step Runge Kutta.
*/
template<typename ForceGenerator>
struct runge_kutta_1 :public thrust::unary_function<real7, real6> {
  ForceGenerator _f;
  real _dt;
  runge_kutta_1(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real7 &in){
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    real m = in.get<6>();
    real3 force = _f(in);
    real3 a = force / m;
    real3 new_p = p + _dt * c; // is this correct without 0.5 * m * a^2 ? 
    real3 new_c = c + _dt * a;
    return detail::make_real6(new_p, new_c);
  }
};
} // END detail

template<typename ForceGenerator>
detail::runge_kutta_1<ForceGenerator> runge_kutta_1(ForceGenerator f, real dt){
  return detail::runge_kutta_1<ForceGenerator>(f, dt);
}

namespace detail {

/*
  2 step Runge Kutta.
*/
template<typename ForceGenerator>
struct runge_kutta_2 :public thrust::unary_function<real7, real6> {
  ForceGenerator _f;
  real _dt;
  runge_kutta_2(ForceGenerator f, real dt)
  :_f(f), _dt(dt){}
  __device__ __host__
  real6 operator()(const real7 &in){
    real3 p = real3(in.get<0>(), in.get<1>(), in.get<2>()); 
    real3 c = real3(in.get<3>(), in.get<4>(), in.get<5>());
    real m = in.get<6>();
    /*
      calc result at half time.
    */
    real6 h_result = bphcuda::runge_kutta_1(_f, 0.5 * _dt)(in);
    /*
      produce state of particle at the time half.
    */
    real7 h_state = make_real7(h_result, m);
    /*
      calc c and a at half time.
    */
    real3 h_c = real3(h_state.get<3>(), h_state.get<4>(), h_state.get<5>());
    real3 h_a = _f(h_state) / m;
    real3 new_p = p + _dt * h_c; // is this correct ? 
    real3 new_c = c + _dt * h_a;
    return detail::make_real6(new_p, new_c);
  }
};
} // END detail

template<typename ForceGenerator>
detail::runge_kutta_2<ForceGenerator> runge_kutta_2(ForceGenerator f, real dt){
  return detail::runge_kutta_2<ForceGenerator>(f, dt);
}

} // END bphcuda
