#pragma once

#include <thrust/random.h>

#include <thrusting/real.h>
#include <thrusting/random/generate.h>
#include <thrusting/functional.h>

#include <bphcuda/cell.h>

namespace {
  using namespace thrust;
  using namespace thrusting;
}

namespace bphcuda {

/*
  generate uniform random number [lower, upper] to array
*/
template<typename Real>
void alloc_uniform_random(
  size_t n_particle,
  Real begin,
  thrusting::real2 range, // min, max
  size_t seed
){
  thrusting::generate(
    begin,
    thrusting::advance(n_particle, begin),
    compose(
      make_uniform_real_distribution<real>(get<0>(range), get<1>(range)),
      make_rng_generator(seed)));     
}

template<typename Real>
void alloc_uniform_random(
  const cell &c,
  size_t n_particle,
  Real x,
  Real y,
  Real z,
  size_t seed
){
  real2 x_range(c.x_min(), c.x_max());
  alloc_uniform_random(
    n_particle,
    x,
    x_range,
    seed + 0);

  real2 y_range(c.y_min(), c.y_max());
  alloc_uniform_random(
    n_particle,
    y,
    y_range,
    seed + 1);
  
  real2 z_range(c.z_min(), c.z_max());
  alloc_uniform_random(
    n_particle,
    z,
    z_range,
    seed + 2);
}

} // END bphcuda
