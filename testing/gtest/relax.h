#pragma once

#include <thrust/iterator/constant_iterator.h>

#include <bphcuda/relax.h>
#include <bphcuda/total_e.h>

#include <thrusting/algorithm/equal.h>
#include <thrusting/real.h>
#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/algorithm/equal.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

// test is imcomplete
TEST(Relax, Test){
  real s = 2; 
  real m = 2; // so that 1/2 * m = 1
  thrust::constant_iterator<real> m_it(m);

  size_t n_particle = 3;
  size_t n_cell = 4;
  
  real _u[] = {1, 1, 1}; vector<real>::type u(_u, _u+n_particle);
  real _v[] = {1, 1, 1}; vector<real>::type v(_v, _v+n_particle);
  real _w[] = {1, 1, 1}; vector<real>::type w(_w, _w+n_particle);
  real _in_e[] = {17, 4, 10}; vector<real>::type in_e(_in_e, _in_e+n_particle);
  
  /*
    total_e is = 
    20 for cellidx 1, number of particle is particularly one in cell 1
    20 for cellidx 2, np = 2
  */
  size_t _idx[] = {1,2,2}; vector<size_t>::type idx(_idx, _idx+n_particle);

  real total_e_1_before = calc_total_e(1, u.begin(), v.begin(), w.begin(), m_it, in_e.begin());
  real total_e_2_before = calc_total_e(2, u.begin()+1, v.begin()+1, w.begin()+1, m_it+1, in_e.begin()+1);
   
  vector<real>::type tmp1(n_cell);
  vector<real>::type tmp2(n_cell);
  vector<real>::type tmp3(n_cell);
  vector<real>::type tmp4(n_cell);
  vector<size_t>::type tmp5(n_cell);
  vector<size_t>::type tmp6(n_cell);

  /*
    relax 
  */
  relax(
    n_particle,
    u.begin(), v.begin(), w.begin(),
    m,
    in_e.begin(),
    s,
    idx.begin(),
    n_cell,
    tmp1.begin(), tmp2.begin(), tmp3.begin(), tmp4.begin(),
    tmp5.begin(), tmp6.begin(),
    7);

  /*
    conserving the total energy 
  */
  real total_e_1_after = calc_total_e(1, u.begin(), v.begin(), w.begin(), m_it, in_e.begin());
  real total_e_2_after = calc_total_e(2, u.begin()+1, v.begin()+1, w.begin()+1, m_it+1, in_e.begin()+1);

  EXPECT_EQ(total_e_1_before, total_e_1_after);
  EXPECT_EQ(total_e_2_before, total_e_2_after);

  /*
    Nothing happened in cell 1
    because there is only one particle in the cell
    and therefore no collision will happen.
  */
  EXPECT_EQ(
    real3(1,1,1), 
    iterator_value_at(
      0, 
      thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin())));
  
  /* 
    assert the in_e is unique for each cell
    note that the cell 1 did not relax.
  */
  real _ans_in_e[] = {17, 4, 4}; vector<real>::type ans_in_e(_ans_in_e, _ans_in_e+n_particle);
  EXPECT_EQ(
    make_list(ans_in_e),
    make_list(in_e));
}
