#pragma once
typedef thrust::tuple<Real, Real> Real2;

typedef thrust::tuple<Real, Real, Real> Real3;

typedef thrust::tuple<Real, Real, Real, Real> Real4;

typedef thrust::tuple<Real, Real, Real, Real, Real> Real5;

typedef thrust::tuple<Real, Real, Real, Real, Real, Real> Real6;

typedef thrust::tuple<Real, Real, Real, Real, Real, Real, Real> Real7;

typedef thrust::tuple<Real, Real, Real, Real, Real, Real, Real, Real> Real8;

typedef thrust::tuple<Real, Real, Real, Real, Real, Real, Real, Real, Real> Real9;

__host__ __device__
Real2 mk_real2(Real x1, Real x2){
  return thrust::make_tuple(x1, x2);
}

__host__ __device__
Real3 mk_real3(Real x1, Real x2, Real x3){
  return thrust::make_tuple(x1, x2, x3);
}

__host__ __device__
Real4 mk_real4(Real x1, Real x2, Real x3, Real x4){
  return thrust::make_tuple(x1, x2, x3, x4);
}

__host__ __device__
Real5 mk_real5(Real x1, Real x2, Real x3, Real x4, Real x5){
  return thrust::make_tuple(x1, x2, x3, x4, x5);
}

__host__ __device__
Real6 mk_real6(Real x1, Real x2, Real x3, Real x4, Real x5, Real x6){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6);
}

__host__ __device__
Real7 mk_real7(Real x1, Real x2, Real x3, Real x4, Real x5, Real x6, Real x7){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7);
}

__host__ __device__
Real8 mk_real8(Real x1, Real x2, Real x3, Real x4, Real x5, Real x6, Real x7, Real x8){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8);
}

__host__ __device__
Real9 mk_real9(Real x1, Real x2, Real x3, Real x4, Real x5, Real x6, Real x7, Real x8, Real x9){
  return thrust::make_tuple(x1, x2, x3, x4, x5, x6, x7, x8, x9);
}
