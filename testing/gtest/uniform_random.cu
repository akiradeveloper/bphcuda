#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <bphcuda/random/uniform_random.h>

#include <iostream>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using thrusting::real2;
}

TEST(uniform_random, printout){
  size_t count = 10;
  THRUSTING_VECTOR<real> xs(count);
  size_t seed = 0;
  bphcuda::alloc_uniform_random(
    count,
    xs.begin(),
    real2(0.1, 0.5),
    seed);
  std::cout << thrusting::make_list(xs) << std::endl;
}
