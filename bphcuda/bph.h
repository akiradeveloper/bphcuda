#pragma once

#include "relax.h"
#include "velocity.h"

namespace {
  using namespace thrusting;
}

namespace bphcuda {

/*
  BPH algorithm.
  particles are already sorted by cell idx.
*/
template<typename Real, typename Int1, typename Int2>
void bph (
  size_t n_particle,
  Real x, Real y, Real z,
  Real u, Real v, Real w,
  real m,
  Real in_e,
  real s,
  Int1 idx,
  size_t n_cell,
  Real tmp1, Real tmp2, Real tmp3, Real tmp4, Real tmp5,
  Real tmp6, Real tmp7, Real tmp10, Real tmp11, Real tmp12,
  Int2 tmp8, Int2 tmp9,
  size_t seed
){
  /*
    minus velocity
  */
  std::cout << "minus" << std::endl;
  minus_average_velocity(
    n_particle,
    u, v, w,
    idx,
    n_cell,
    tmp1, tmp2, tmp3, // ave_c
    tmp4, tmp5, tmp6, // tmp
    tmp8, tmp9); // tmp
     
  /*
    relaxing
  */
  std::cout << "relax" << std::endl;
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

  /*
    back the velocity
  */
  std::cout << "plus" << std::endl;
  plus_average_velocity(
    n_particle,
    u, v, w,
    idx,
    n_cell,
    tmp1, tmp2, tmp3); // ave_c
}

} // END bphcuda
