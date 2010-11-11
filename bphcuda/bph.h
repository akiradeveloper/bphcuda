#pragma once

#include <thrusting/dtype/real.h>

#include <bphcuda/relaxing.h>
#include <bphcuda/share_e.h>

namespace {
  using thrusting::real;
}

namespace bphcuda {

template<typename R>
void bph(
  size_t n_particle,
  R u, R v, R w,
  real m, // m is constant shared by all particles
  size_t seed
){
  relax(n_particle, u, v, w, seed);
}

template<typename R>
void bph(
  size_t n_particle,
  R u, R v, R w,
  real m,
  R in_e,
  real s,
  size_t seed
){
  relax(n_particle, u, v, w, seed);
  share_e(n_particle, u, v, w, m, in_e, s);
}

} // END bphcuda
