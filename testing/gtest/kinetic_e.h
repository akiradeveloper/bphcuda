#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrusting/vector.h>

#include <bphcuda/kinetic_e.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(KineticE, Test){
  real e = calc_kinetic_e(
    1, // n_particle
    thrust::make_counting_iterator<real>(-1),    
    thrust::make_counting_iterator<real>(2),    
    thrust::make_counting_iterator<real>(3),    
    thrust::make_counting_iterator<real>(2)); // m

  EXPECT_EQ(14.0, e);
}
