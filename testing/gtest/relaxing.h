#include <iostream>

#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <bphcuda/relaxing.h>
#include <bphcuda/momentum.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
}

/*
  Akira Hayakawa 2010 11/13
  This Test is completely wrong.
  Misunderstanding of physics.
  The energy, momentum are preserved
  only in gravity center system.
*/

// case 3 particles 
TEST(relaxing, n_particle_even){
  size_t n_particle = 3;
  real _us[] = {1.0, 4.0, 7.0}; THRUSTING_VECTOR<real> us(_us, _us+n_particle);
  real _vs[] = {2.0, 5.0, 8.0}; THRUSTING_VECTOR<real> vs(_vs, _vs+n_particle);
  real _ws[] = {3.0, 6.0, 9.0}; THRUSTING_VECTOR<real> ws(_ws, _ws+n_particle);

  real mass = 1.0;

  real3 old_momentum =
    bphcuda::calc_momentum(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(mass));

  real old_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(mass));
  
  size_t seed = 0;
  bphcuda::relax(
    n_particle,
    us.begin(),
    vs.begin(),
    ws.begin(),
    seed);

  real3 new_momentum =
    bphcuda::calc_momentum(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(mass));
    
  real new_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(mass));
  
  // the last element is zero speed
  EXPECT_EQ(
    real3(0.0, 0.0, 0.0),
    thrusting::iterator_value_at(2, thrusting::make_zip_iterator(us.begin(), vs.begin(), ws.begin())));

  // preserving the momentum
  EXPECT_EQ(old_momentum, new_momentum);

  // preserving the energy
  EXPECT_EQ(old_kinetic_e, new_kinetic_e);
}

// case 2 particles
TEST(relaxing, n_particle_odd){
}
