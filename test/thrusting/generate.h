#pragma once

#include <thrusting/random/generate.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/functional.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Generate, Test){
  vector<float>::type output(10);
 
  thrusting::generate(
    output.begin(),
    output.end(),
    compose(
      make_uniform_real_distribution<float>(0, 10),
      make_fast_rng_generator(777)));

  std::cout << make_list(output) << std::endl;
}
