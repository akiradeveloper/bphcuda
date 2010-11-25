#pragma once

#include "relax_cell_parallel.h"
#include "cell.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<
typename R1, 
typename R2,
typename R3>
void bph_cell_parallel(
  cell c,
  size_t n_particle,
  R1 x, R1 y, R1 z,
  R2 u, R2 v, R2 w,
  real m,
  R3 in_e,
  real s,
  // ...
  size_t seed
){
  // subtract the velocity of gravity point

  // sorting

  // relaxing by each cell
  for(size_t i=0; i<n_cell; i++){
    relax_cell_parallel();
  }

  // back the velocity
}

} // END bphcuda
