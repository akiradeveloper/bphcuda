#pragma once

#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

namespace {
  using thrusting::real3;
  typedef thrust::tuple<size_t, size_t, size_t> size3;
}

namespace bphcuda {

struct cell {
  real3 origin;
  real3 spaces;
  size3 dims;
  real x_min(){
    return origin.get<0>();
  }  
  real x_max(){
    return origin.get<0>() + dims.get<0>() * spaces.get<0>();
  }
  real y_min(){
    return origin.get<1>();
  }  
  real y_max(){
    return origin.get<1>() + dims.get<1>() * spaces.get<1>();
  }
  real z_min(){
    return origin.get<2>();
  }  
  real z_max(){
    return origin.get<2>() + dims.get<2>() * spaces.get<2>();
  }
};

__host__ __device__
cell make_cell(
real3 origin,
real3 spaces,
size3 dims){
  cell c;
  c.origin = origin;
  c.spaces = spaces;
  c.dims = dims;
  return c;
}

__host__ __device__
size3 calc_idx3(const cell &c, const real3 &p){
  size_t xidx = (p.get<0>()-c.origin.get<0>()) / c.spaces.get<0>();
  size_t yidx = (p.get<1>()-c.origin.get<1>()) / c.spaces.get<1>();
  size_t zidx = (p.get<2>()-c.origin.get<2>()) / c.spaces.get<2>();
  return size3(xidx, yidx, zidx);
}

__host__ __device__
size_t conv_idx3_idx1(const cell &c, const size3 &idx3){
  return idx3.get<0>() * c.dims.get<1>() * c.dims.get<2>() +
         idx3.get<1>() * c.dims.get<2>() +
         idx3.get<2>();
}

__host__ __device__
size_t calc_idx1(const cell &c, const real3 &p){
  return conv_idx3_idx1(c, calc_idx3(c, p));
}

/*
  p -> idx
*/
struct calc_cellidx1 :public thrust::unary_function<real3, size_t> {
  cell c;
  calc_cellidx1(cell c_)
  :c(c_){}
  __host__ __device__
  size_t operator()(const real3 &p) const {
    return calc_idx1(c, p);
  }
};

} // END bphcuda
