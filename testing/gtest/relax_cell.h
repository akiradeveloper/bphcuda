#pragma once
#include <iostream>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <bphcuda/relaxing.h>
#include <bphcuda/momentum.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

// test is imcomplete
TEST(RelaxCell, Test){
  size_t n_particle = 3;
  real _us[] = {1.0, 4.0, 7.0}; vector<real>::type us(_us, _us+n_particle);
  real _vs[] = {2.0, 5.0, 8.0}; vector<real>::type vs(_vs, _vs+n_particle);
  real _ws[] = {3.0, 6.0, 9.0}; vector<real>::type ws(_ws, _ws+n_particle);
  real _in_e[] = 

  real m = 2;

  size_t seed = 0;
  bphcuda::relax_cell(
    n_particle,
    us.begin(),
    vs.begin(),
    ws.begin(),
    in_e.begin(),
    seed);

  real3 new_momentum =
    bphcuda::calc_momentum(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(m));
    
  real new_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      us.begin(),
      vs.begin(),
      ws.begin(),
      thrust::constant_iterator<real>(m));
  
  /*
    preserving the total energy
  */

  /*
    momentum is zero after all
  */

  /*
    total_kinetic_e : total_in_e = 3:s
  */
}

