#pragma once

#include <bphcuda/real.h>
#include <bphcuda/int.h>

namespace bphcuda {

struct Cell {
  Real3 origin;
  Real3 spaces;
  Int3 dims;
};
  
Cell mk_cell(Real3 origin, Real3 spaces, Int3 dims){
  Cell c;
  c.origin = origin;
  c.spaces = spaces;
  c.dims = dims;
}

Int3 calc_ind3(const Cell &c, const Real3 p){
  Int xind = (p.x-c.origin.x) / c.spaces.x;
  Int yind = (p.y-c.origin.y) / c.spaces.y;
  Int zind = (p.z-c.origin.z) / c.spaces.z;
  return mk_int3(xind, yind, zind);
}

Int conv_ind3_ind1(const Cell &c, const &ind3){
  return ind3.x * c.dims.y * c.dims.z +
         ind3.y * c.dims.z +
         ind3.z;
}

Int calc_ind1(const Cell &c, const Real3 p){
  return conv_ind3_ind1(c, calc_ind3(c, p));
}

struct calc_cellind1 :public thrust::unary_function<Real3, Int> {
  Cell c;
  calc_cellind1(Cell c_)
  :c(c_){}
  Int operator()(Real3 p){
    return calc_ind1(c, p);
  }
};

} // end of bphcuda
