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
dim3 calc_idx3(const cell &c, const real3 &p){
  size_t xidx = (p.get<0>()-c.origin.get<0>()) / c.spaces.get<0>();
  size_t yidx = (p.get<1>()-c.origin.get<1>()) / c.spaces.get<1>();
  size_t zidx = (p.get<2>()-c.origin.get<2>()) / c.spaces.get<2>();
  return thrustin::make_tuple3<size_t>(xidx, yidx, zidx);
}

__host__ __device__
size_t conv_idx3_idx1(const cell &c, const dim3 &idx3){
  return idx3.get<0>() * c.dims.get<1>() * c.dims.get<2>() +
         idx3.get<1>() * c.dims.get<2>() +
         idx3.get<2>();
}

__host__ __device__
size_t calc_idx1(const cell &c, const real3 &p){
  return conv_idx3_idx1(c, calc_idx3(c, p));
}

struct calc_cellidx1 :public thrust::unary_function<real3, size_t> {
  Cell c;
  calc_cellidx1(Cell c_)
  :c(c_){}
  __host__ __device__
  size_t operator()(const real3 &p) const {
    return calc_idx1(c, p);
  }
};

} // END bphcuda
