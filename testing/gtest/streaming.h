#include <bphcuda/streaming.h>

#include <thrusting/real.h>
#include <thrusting/functional.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
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
  
  real6 after = bphcuda::runge_kutta_1(
    thrusting::constant(real3(-1,0,0)), 0.2)(before);
    
   
  real3 pos = real3(after.get<0>(), after.get<1>(), after.get<2>());
  real3 vel = real3(after.get<3>(), after.get<4>(), after.get<5>());

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
  
  real6 after = bphcuda::runge_kutta_2(
    thrusting::constant(real3(-1,0,0)), 0.2)(before);
    
  real3 pos = real3(after.get<0>(), after.get<1>(), after.get<2>());
  real3 vel = real3(after.get<3>(), after.get<4>(), after.get<5>());

  EXPECT_EQ(
    real3(0.18,0,0),
    pos);

  EXPECT_EQ(
    real3(0.8,0,0),
    vel);
}
