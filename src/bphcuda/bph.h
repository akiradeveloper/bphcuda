#pragma once

#include <iostream>

#include "relax.h"
#include "velocity.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

/*
 * The facade of BPH algorithm.
 * Requiring particles are already sorted by cell idx.
 */
template<typename Real, typename Int1, typename Int2>
void bph (
  size_t n_particle,
  Real x, Real y, Real z, // position
  Real u, Real v, Real w, // velocity in center of gravity system
  real m, // mass of a particle. mass must be unique.
  Real in_e, // internal energy
  real s, // internal degree of freedom of a particle. This quantity must also be unique.
  Int1 idx, // The indices of particles. Must be sorted. 
  size_t n_cell, 
  /*
   * Needs
   * 11 temporary vector of real type and
   * 2 for integer type.
   */
  Real tmp1, Real tmp2, Real tmp3, Real tmp4, Real tmp5,
  Real tmp6, Real tmp7, Real tmp10, Real tmp11, Real tmp12,
  Int2 tmp8, Int2 tmp9,
  size_t seed // seed is needed for randomness.
){
  minus_average_velocity(
    n_particle,
    u, v, w,
    idx,
    n_cell,
    tmp1, tmp2, tmp3, // ave_c
    tmp4, tmp5, tmp6, // tmp
    tmp8, tmp9); // tmp
     
  /*
   * relaxing
   */
  relax(
    n_particle,
    u, v, w,
    m,
    in_e,
    s,
    idx,
    n_cell,
    tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, // tmp
    tmp8, tmp9, // tmp
    seed);

  plus_average_velocity(
    n_particle,
    u, v, w,
    idx,
    n_cell,
    tmp1, tmp2, tmp3); // ave_c
}

} // END bphcuda
