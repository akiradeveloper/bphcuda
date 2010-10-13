#pragma once

#include <thrust/tuple.h>

namespace bphcuda {

typedef float Real; // this means nothing in fact
typedef thrust::tuple<Real, Real, Real> Real3; // conceal Real3 is float3 or float4 for coarescing

__host__ __device__
Real3 mk_real3(Real x, Real y, Real z){
  return thrust::make_tuple(x, y, z);
}

Real3 operator*(Real3 p, Real val){
  Real x = val * p.get<0>();
  Real y = val * p.get<1>();
  Real z = val * p.get<2>();
  return mk_real3(x, y, z);
}

Real3 operator*(Real3 p1, Real3 p2){
  Real x = p1.get<0>() * p2.get<0>();
  Real y = p1.get<1>() * p2.get<1>();
  Real z = p1.get<2>() * p2.get<2>();
  return mk_real3(x, y, z);
}

Real3 operator+(Real3 p1, Real3 p2){
  Real x = p1.get<0>() + p2.get<0>();
  Real y = p1.get<1>() + p2.get<1>();
  Real z = p1.get<2>() + p2.get<2>();
  return mk_real3(x, y, z);
}

} // end of bphcuda
