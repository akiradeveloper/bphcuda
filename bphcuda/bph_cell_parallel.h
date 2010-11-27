#pragma once

#include "relax_cell_parallel.h"
#include "cell.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<typename Real>
void bph_cell_parallel(
  cell c,
  size_t n_particle,
  Real x, Real y, Real z,
  Real u, Real v, Real w,
  real m,
  Real in_e,
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
