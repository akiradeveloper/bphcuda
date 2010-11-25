#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <thrust/iterator/constant_iterator.h>

#include <bphcuda/share_e.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

/*
  Total energy is 20 of 3:17.
  and it will be shared to 12:8,
  where velocity is (2,2,2).
*/
TEST(ShareE, Test){
  size_t n_particle = 1;
  real _us[] = {1}; vector<real>::type us(_us, _us+n_particle);
  real _vs[] = {1}; vector<real>::type vs(_vs, _vs+n_particle);
  real _ws[] = {1}; vector<real>::type ws(_ws, _ws+n_particle);
  real mass = 2;
  real _in_e[] = {17}; vector<real>::type in_e(_in_e, _in_e+n_particle);
  real s = 2; // will be shared by 3:2
  
  bphcuda::share_e(
    n_particle,
    us.begin(),
    vs.begin(),
    ws.begin(),
    thrust::constant_iterator<real>(mass),
    in_e.begin(),
    s);
  
  EXPECT_EQ(
    real3(2.0, 2.0, 2.0), 
    thrusting::iterator_value_at(0, thrusting::make_zip_iterator(us.begin(), vs.begin(), ws.begin())));

  EXPECT_EQ(
    real(8.0),
    thrusting::iterator_value_at(0, in_e.begin()));
}
