#pragma once

#include <thrust/iterator/constant_iterator.h>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/iterator.h>
#include <thrusting/algorithm/equal.h>

#include <bphcuda/alloc_in_e.h>
#include <bphcuda/real_comparator.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(AllocInE, Test){
  real s = 2; 
  real m = 2; // so that 1/2 * m = 1

  size_t n_particle = 3;
  size_t n_cell = 4;
  
  real _u[] = {1, 4, 7}; vector<real>::type u(_u, _u+n_particle);
  real _v[] = {2, 5, 8}; vector<real>::type v(_v, _v+n_particle);
  real _w[] = {3, 6, 9}; vector<real>::type w(_w, _w+n_particle);
  
  /*
    Arbitrary in_e initially.
  */
  real _in_e[] = {777, 888, 999}; vector<real>::type in_e(_in_e, _in_e+n_particle);
  
  size_t _idx[] = {1,1,2}; vector<size_t>::type idx(_idx, _idx+n_particle);

  vector<real>::type tmp1(n_cell);
  vector<real>::type tmp4(n_cell);
  vector<size_t>::type tmp2(n_cell);
  vector<size_t>::type tmp3(n_cell);

  bphcuda::alloc_in_e(
    n_particle,
    u.begin(), v.begin(), w.begin(),
    m,
    in_e.begin(),
    s,
    idx.begin(),
    n_cell,
    tmp1.begin(), tmp4.begin(), // real
    tmp2.begin(), tmp3.begin()); // int

  real e1 = real(91) / 3;
  real e2 = real(194) * 2 / 3;

  real _ans_in_e[] = {e1, e1, e2}; vector<real>::type ans_in_e(_ans_in_e, _ans_in_e+n_particle);
  EXPECT_TRUE(
    thrusting::equal(
      n_particle,
      ans_in_e.begin(),
      in_e.begin(),
      make_real_comparator(1, 0.0001)));
}
