#pragma once

#include <bphcuda/value.h>

#include <thrust/functional.h>

namespace bphcuda {

struct rand_shell :public thrust::unary_function<thrust::tuple<Real, Real>, Real3> {
  __host__ __device__
  Real operator()(thrust::tuple<Real, Real> rands){
  }
};

typedef thrust::tuple<Real, Real, Real, Real, Real, Real> Real6;
struct rand_shell :public thrust::unary_function<Real6, Real3> {
  Real T;
  Real m;
  rand_shell(Real T_, Real m_)
  :T(T_), m(m_){}
  __host__ __device__
  Real3 operator()(Real6 rands){
  }
};

} // end of bph
