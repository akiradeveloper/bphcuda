#pragma once

#include <thrusting/random/distribution.h>

#include <iostream>
#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Distribution, UniformReal){
  thrust::default_random_engine rng(777);
  float x = make_uniform_real_distribution<float>(0.5, 0.8)(rng);
  EXPECT_GE(x, 0.5);
  EXPECT_LT(x, 0.8);
}

TEST(Distribution, UniformInt){
  thrust::default_random_engine rng(777);
  int x = make_uniform_int_distribution<int>(2, 5)(rng);
  EXPECT_GE(x, 2);
  EXPECT_LT(x, 5);
}
