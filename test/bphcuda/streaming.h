#pragma once

#include <bphcuda/streaming.h>
#include <bphcuda/real_comparator.h>

#include <thrusting/real.h>
#include <thrusting/functional.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrust;
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(Streaming, Detail){
  EXPECT_EQ(real6(1,2,3,4,5,6), bphcuda::detail::make_real6(real3(1,2,3), real3(4,5,6)));
  EXPECT_EQ(real7(1,2,3,4,5,6,7), bphcuda::detail::make_real7(real6(1,2,3,4,5,6), 7));
}

TEST(RungeKutta1, ConstantForce){
  real7 before = real7(
    0,0,0,
    1,0,0,
    1); 
  
  real6 after = bphcuda::make_runge_kutta_1_functor(
    thrusting::make_constant_functor<real7>(real3(-1,0,0)), 0.2)(before);
    
  real3 pos = real3(get<0>(after), get<1>(after), get<2>(after));
  real3 vel = real3(get<3>(after), get<4>(after), get<5>(after));

  EXPECT_EQ(
    real3(0.2,0,0),
    pos);

  EXPECT_EQ(
    real3(0.8,0,0),
    vel);
}

TEST(RungeKutta2, ConstantForce){
  real7 before = real7(
    0,0,0,
    1,0,0,
    1); 
  
  real6 after = bphcuda::make_runge_kutta_2_functor(
    thrusting::make_constant_functor<real7>(real3(-1,0,0)), 0.2)(before);
    
  real3 pos = real3(get<0>(after), get<1>(after), get<2>(after));
  real3 vel = real3(get<3>(after), get<4>(after), get<5>(after));

  EXPECT_TRUE(make_real3_comparator(real3(1,1,1), 0.0001)
  (real3(0.18, 0, 0), pos));
  
  EXPECT_TRUE(make_real3_comparator(real3(1,1,1), 0.0001)
  (real3(0.8, 0, 0), vel));
}
