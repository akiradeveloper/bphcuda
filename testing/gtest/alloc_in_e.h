#include <thrust/iterator/constant_iterator.h>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/iterator.h>

#include <bphcuda/alloc_in_e.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(AllocInE, Test){
  real s = 2; 
  real m = 2; // so that 1/2 * m = 1

  size_t n_particle = 3;
  size_t n_cell = 4;
  
  real _u[] = {1, 4, 7}; vector<real>::type u(_u, _u+n_particle);
  real _v[] = {2, 5, 8}; vector<real>::type v(_v, _v+n_particle);
  real _w[] = {3, 6, 9}; vector<real>::type w(_w, _w+n_particle);
  real _in_e[] = {100, 200, 300}; vector<real>::type in_e(_in_e, _in_e+n_particle);
  
  size_t _idx[] = {1,2,2}; vector<size_t>::type idx(_idx, _idx+n_particle);

  vector<real>::type tmp1(n_cell);
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
    tmp1.begin(),
    tmp2.begin(),
    tmp3.begin());

  real e1 = real(91) / 3;
  real e2 = real(194) * 2 / 3;

  real _ans_in_e[] = {e1, e1, e2}; vector<real>::type ans_in_e(_ans_in_e, _ans_in_e+n_particle);
  EXPECT_EQ(
    make_list(ans_in_e),
    make_list(in_e));
}
