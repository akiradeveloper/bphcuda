#pragma once

#include <iostream>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <bphcuda/relax_cell.h>
#include <bphcuda/momentum.h>
#include <bphcuda/kinetic_e.h>
#include <bphcuda/total_e.h>
#include <bphcuda/real_comparator.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

// test is imcomplete
TEST(RelaxCell, Test){
  size_t n_particle = 3;
  real m = 2;
  real s = 2;

  /*
    total_e is 60 
  */
  real _us[] = {1.0, 2.0, 3.0}; vector<real>::type us(_us, _us+n_particle);
  real _vs[] = {1.0, 2.0, 3.0}; vector<real>::type vs(_vs, _vs+n_particle);
  real _ws[] = {1.0, 2.0, 3.0}; vector<real>::type ws(_ws, _ws+n_particle);
  real _in_e[] = {3, 6, 9}; vector<real>::type in_e(_in_e, _in_e+n_particle);

  real old_total_e =
    bphcuda::calc_total_e(
      n_particle,
      us.begin(), vs.begin(), ws.begin(),
      thrust::make_constant_iterator(m),
      in_e.begin());

  size_t seed = 777;
  bphcuda::relax_cell(
    n_particle,
    us.begin(), vs.begin(), ws.begin(),
    m,
    in_e.begin(),
    s,
    seed);

  real3 new_total_momentum =
    bphcuda::calc_momentum(
      n_particle,
      us.begin(), vs.begin(), ws.begin(),
      thrust::constant_iterator<real>(m));
    
  real new_total_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      us.begin(), vs.begin(), ws.begin(),
      thrust::constant_iterator<real>(m));

  real new_total_in_e =
    thrust::reduce(in_e.begin(), in_e.end());

  real new_total_e = new_total_kinetic_e + new_total_in_e;
  
  /*
    preserving the total energy
  */
  EXPECT_TRUE(
    make_real_comparator(1, 0.0001)(old_total_e, new_total_e));

  /*
    momentum is zero after all
  */
  EXPECT_TRUE(
    make_real3_comparator(real3(1,1,1), 0.0001)(real3(0,0,0), new_total_momentum));

  /*
    total_kinetic_e : total_in_e = 3:s
  */
  EXPECT_TRUE(
    make_real_comparator(1, 0.0001)(new_total_kinetic_e / new_total_in_e, 3/s));
}

