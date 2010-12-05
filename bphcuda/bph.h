#pragma once

#include "relax_particle_parallel.h"
#include "cell.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

template<typename Real, typename Int>
void bph (
  const cell &c,
  size_t n_particle,
  Real x, Real y, Real z,
  Real u, Real v, Real w,
  real m,
  Real in_e,
  real s,
  size_t n_cell,
  Real tmp, Real tmp, Real tmp, Real tmp,
  Int tmp, Int tmp,
  size_t seed
){
  // minus velocity

  // sorting

  // relaxing
  relax_particle_parallel();

  // back the velocity
}

} // END bphcuda
