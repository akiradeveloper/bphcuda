#include <bphcuda/force.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Force, NoForce){
  EXPECT_EQ(real3(0,0,0), bphcuda::no_force()(real7(1,2,3,4,5,6,7))); 
}

TEST(Force, Pow){
  EXPECT_EQ(real(125), pow(25, real(1.5)));
}

TEST(Force, GravitationalForce){
  real3 P = real3(0,0,0);
  real M = 1;
  real G = 1;

  real7 obj = real7(
    1,0,0, // p
    0,0,0, // c
    1); // m
   
  std::cout << obj << std::endl;

  EXPECT_EQ(
    real3(-1,0,0),
    bphcuda::gravitational_force(
      P,
      M,
      G)(obj));
}
