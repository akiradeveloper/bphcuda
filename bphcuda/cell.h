#pragma once

#include <bphcuda/real.h>
#include <bphcuda/int.h>
#include <thrust/functional.h>

namespace bphcuda {

struct Cell {
  Real3 origin;
  Real3 spaces;
  Int3 dims;
};
  
__host__ __device__
Cell mk_cell(Real3 origin, Real3 spaces, Int3 dims){
  Cell c;
  c.origin = origin;
  c.spaces = spaces;
  c.dims = dims;
  return c;
}

__host__ __device__
Int3 calc_ind3(const Cell &c, const Real3 &p){
  Int xind = (p.get<0>()-c.origin.get<0>()) / c.spaces.get<0>();
  Int yind = (p.get<1>()-c.origin.get<1>()) / c.spaces.get<1>();
  Int zind = (p.get<2>()-c.origin.get<2>()) / c.spaces.get<2>();
  return mk_int3(xind, yind, zind);
}

__host__ __device__
Int conv_ind3_ind1(const Cell &c, const Int3 &ind3){
  return ind3.get<0>() * c.dims.get<1>() * c.dims.get<2>() +
         ind3.get<1>() * c.dims.get<2>() +
         ind3.get<2>();
}

__host__ __device__
Int calc_ind1(const Cell &c, const Real3 &p){
  return conv_ind3_ind1(c, calc_ind3(c, p));
}

struct calc_cellind1 :public thrust::unary_function<Real3, Int> {
  Cell c;
  calc_cellind1(Cell c_)
  :c(c_){}
  __host__ __device__
  Int operator()(const Real3 &p){
    return calc_ind1(c, p);
  }
};

} // end of bphcuda
