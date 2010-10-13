#pragma once

#include <bphcuda/real.h>
#include <bphcuda/int.h>

namespace bphcuda {

struct Cell {
  Real3 origin;
  Real3 spaces;
  Int3 divisions;
}
  
Cell mk_cell(Real3 origin, Real3 spaces, Int3 divisions){
  Cell c;
  c.origin = origin;
  c.spaces = spaces;
  c.divisions = divisions;
}

Int3 calc_ind3(const Cell &c, const Real3 p){
}

Int conv_ind3_ind1(const Cell &c, const &ind3){
}

} // end of bphcuda
