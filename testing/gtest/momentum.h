#include <thrust/iterator/constant_iterator.h>

#include <thrusting/real.h>

#include <bphcuda/momentum.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Momentum, Test){
  size_t n_particle = 10;
  real m = 1.0;
  
  real3 momentum = bphcuda::calc_momentum(
    n_particle,
    thrust::constant_iterator<real>(1.0),
    thrust::constant_iterator<real>(2.0),
    thrust::constant_iterator<real>(3.0),
    thrust::constant_iterator<real>(m));
  
  EXPECT_EQ(real3(10.0, 20.0, 30.0), momentum);
}
