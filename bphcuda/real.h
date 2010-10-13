#pragma once

namespace bphcuda {

typedef float Real // this means nothing in fact
typedef thrust::tuple<Real, Real, Real> Real3 // conceal Real3 is float3 or float4 for coarescing

Real3 mk_real3(Real x, Real y, Real z){
  return thrust::make_tuple(x, y, z);
}

Real3 operator*(Real p, Real val){
  Real x = val * p.x;
  Real y = val * p.y;
  Real z = val * p.z;
  return mk_real3(x, y, z);
}

Real3 operator*(Real3 p1, Real3 p2){
  Real x = p1.x * p2.x;
  Real y = P1.y * p2.y;
  Real z = p1.z * p2.z;
  return mk_real3(x, y, z);
}

Real3 operator+(Real3 p1, Real3 p2){
  Real x = p1.x + p2.x;
  Real y = p1.y + p2.y;
  Real z = p1.z + p2.z;
  return mk_real3(x, y, z);
}

} // end of bphcuda
