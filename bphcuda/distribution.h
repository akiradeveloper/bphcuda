#pragma once

#include <bphcuda/real.h>

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace bphcuda {

struct shell_rand :public thrust::unary_function<thrust::tuple<Real, Real>, Real3> {
  __host__ __device__
  Real3 operator()(thrust::tuple<Real, Real> rands){
    Real a = 2 * PI * rand.x;
    Real b = 2 * PI * rand.y;
    Real cx = cosf(a) * cosf(b);
    Real cy = cosf(a) * sinf(b);
    Real cz = sinf(a);
    return mk_real3(cx, cy, cz);
  }
};

template<typename Iter>
void alloc_shell_rand(Iter first, Iter last, Int seed){
}

typedef thrust::tuple<Real, Real, Real, Real, Real, Real> Real6;
struct maxwell_rand :public thrust::unary_function<Real6, Real3> {
  Real T;
  Real m;
  rand_shell(Real T_, Real m_)
  :T(T_), m(m_){}
  __host__ __device__
  Real3 operator()(Real6 rands){
    
  }
};

template<typename Iter>
void alloc_maxwell_rand(

} // end of bph
