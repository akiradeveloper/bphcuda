#pragma once

#include <bphcuda/force.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(Force, NoForce){
  EXPECT_EQ(real3(0,0,0), make_no_force_generator()(real7(1,2,3,4,5,6,7))); 
}

// I think not needed.
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
   
  EXPECT_EQ(
    real3(-1,0,0),
    make_gravitational_force_generator(
      P,
      M,
      G)(obj));
}
