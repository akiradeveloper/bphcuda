#include <bphcuda/force.h>

#include <thrusting/real.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}


TEST(Force, NoForce){
  TEST(real3(0,0,0), no_force()(real7(0,0,0,0,0,0,0))); 
}

TEST(Force, GravitationalForce){
  TEST(real3(0,0,0), gravitational_force(real3, real, real)(real7));
}
