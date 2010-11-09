#pragma once

#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

namespace {
  using thrusting::real3;
  typedef thrust::tuple<size_t, size_t, size_t> dim3;
}

namespace bphcuda {

struct cell {
  real3 origin;
  real3 spaces;
  dim3 dims;
};
  
__host__ __device__
cell make_cell(
real3 origin,
real3 spaces,
dim3 dims){
  cell c;
  c.origin = origin;
  c.spaces = spaces;
  c.dims = dims;
  return c;
}

__host__ __device__
dim3 calc_ind3(const cell &c, const real3 &p){
  size_t xind = (p.get<0>()-c.origin.get<0>()) / c.spaces.get<0>();
  size_t yind = (p.get<1>()-c.origin.get<1>()) / c.spaces.get<1>();
  size_t zind = (p.get<2>()-c.origin.get<2>()) / c.spaces.get<2>();
  return thrustin::make_tuple3<size_t>(xind, yind, zind);
}

__host__ __device__
size_t conv_ind3_ind1(const cell &c, const dim3 &ind3){
  return ind3.get<0>() * c.dims.get<1>() * c.dims.get<2>() +
         ind3.get<1>() * c.dims.get<2>() +
         ind3.get<2>();
}

__host__ __device__
size_t calc_ind1(const cell &c, const real3 &p){
  return conv_ind3_ind1(c, calc_ind3(c, p));
}

struct calc_cellind1 :public thrust::unary_function<real3, size_t> {
  Cell c;
  calc_cellind1(Cell c_)
  :c(c_){}
  __host__ __device__
  size_t operator()(const real3 &p) const {
    return calc_ind1(c, p);
  }
};

} // END bphcuda
