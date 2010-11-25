#pragma once

#include "relax_particle_parallel.h"
#include "cell.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<
typename R1, 
typename R2,
typename R3>
void bph_particle_parallel(
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
  // minus velocity

  // sorting

  // relaxing
  relax_particle_parallel();

  // back the velocity
}

} // END bphcuda
