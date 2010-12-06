#pragma once

#include <thrust/iterator/constant_iterator.h>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/list.h>
#include <thrusting/vectorspace.h>

#include <bphcuda/real_comparator.h>
#include <bphcuda/distribution/maxwell_distribution.h>

#include <iostream>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

TEST(MaxwellDistribution, PrintOut){
  size_t count = 100000;
  vector<real>::type u(count);
  vector<real>::type v(count);
  vector<real>::type w(count);
  real m = 1.0;
  real T = 1.0;
  size_t seed = 0;
  bphcuda::alloc_maxwell_rand(
    count,
    u.begin(), v.begin(), w.begin(), 
    m,
    T,
    seed);

  real3 sum_c = thrust::reduce(
    thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin()),
    thrusting::advance(count, thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin())),
    real3(0,0,0));
 
  std::cout << sum_c << std::endl;
  EXPECT_TRUE(make_real3_comparator(0)(real3(0,0,0), sum_c));
}
