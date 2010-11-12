#include <thrust/iterator/constant_iterator.h>

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

#include <bphcuda/momentum.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
}

TEST(momentum, test1){
  size_t n_particle = 10;
  
  real3 momentum = bphcuda::calc_momentum(
    n_particle,
    thrust::constant_iterator<real>(1.0),
    thrust::constant_iterator<real>(2.0),
    thrust::constant_iterator<real>(3.0),
    thrust::constant_iterator<real>(1.0));  
  
  EXPECT_EQ(real3(10.0, 20.0, 30.0), momentum);
}
