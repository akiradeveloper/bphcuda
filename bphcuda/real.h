#pragma once

#include <thrust/tuple.h>

namespace bphcuda {

typedef float Real;
typedef thrust::tuple<Real, Real, Real> Real3;

__host__ __device__
Real3 mk_real3(Real x, Real y, Real z){
  return thrust::make_tuple(x, y, z);
}

__host__ __device__
Real3 operator*(const Real3 &p1, const Real3 &p2){
  Real x = p1.get<0>() * p2.get<0>();
  Real y = p1.get<1>() * p2.get<1>();
  Real z = p1.get<2>() * p2.get<2>();
  return mk_real3(x, y, z);
}

__host__ __device__
Real3 operator*(const Real3 &p, Real val){
  return p * mk_real3(val, val, val);
}

__host__ __device__
Real3 operator*(Real val, const Real3 &p){
  return p * mk_real3(val, val, val);
}

__host__ __device__
Real3 operator+(const Real3 &p1, const Real3 &p2){
  Real x = p1.get<0>() + p2.get<0>();
  Real y = p1.get<1>() + p2.get<1>();
  Real z = p1.get<2>() + p2.get<2>();
  return mk_real3(x, y, z);
}

} // end of bphcuda
