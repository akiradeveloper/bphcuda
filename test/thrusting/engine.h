#pragma once

#include <thrusting/random/engine.h>
#include <thrusting/random/distribution.h>

#include <thrusting/functional.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Engine, Fast){
  float x = compose(
    make_uniform_real_distribution<float>(3, 5), 
    make_fast_rng_generator(777))(10000);

  EXPECT_GE(x, 3);
  EXPECT_LT(x, 5);
}

TEST(Engine, Discard){
  float x = compose(
    make_uniform_int_distribution<int>(3, 5), 
    make_rng_generator(777))(10000);

  EXPECT_GE(x, 3);
  EXPECT_LT(x, 5);
}
