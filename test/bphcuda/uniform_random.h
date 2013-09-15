#pragma once

#include <thrust/transform_reduce.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/functional.h>

#include <bphcuda/random/uniform_random.h>

//#include <iostream>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(UniformRandom, Test){
  size_t count = 10;
  vector<real>::type xs(count);
  size_t seed = 0;

  bphcuda::alloc_uniform_random(
    count,
    xs.begin(),
    real2(0.1, 0.5),
    seed);
  
  EXPECT_TRUE(thrust::transform_reduce(
    xs.begin(), xs.end(), 
    thrusting::bind2nd(thrust::less_equal<real>(), real(0.5)),
    true,
    thrust::logical_and<bool>()));

  EXPECT_TRUE(thrust::transform_reduce(
    xs.begin(), xs.end(), 
    thrusting::bind2nd(thrust::greater_equal<real>(), real(0.1)),
    true,
    thrust::logical_and<bool>()));
}
