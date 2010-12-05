#pragma once

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include <thrusting/real.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/iterator/zip_iterator.h>

#include <bphcuda/distribution/shell_distribution.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

struct SHELL_DIST_POW :public thrust::unary_function<real3, real> {
  __host__ __device__
  real operator()(const real3 &x) const {
    return x.get<0>() * x.get<0>() + x.get<1>() * x.get<1>() + x.get<2>() * x.get<2>();
  }
}; 

TEST(ShellDistribution, Test){
  size_t count = 5;
  vector<real>::type u(count);
  vector<real>::type v(count);
  vector<real>::type w(count);
  size_t seed = 0;
  bphcuda::alloc_shell_rand(
    count,
    u.begin(), v.begin(), w.begin(), 
    seed);

  EXPECT_EQ(
    make_list(count, thrust::make_constant_iterator(1)),
    make_list(
      count,
      thrust::make_transform_iterator(
        thrusting::make_zip_iterator(u.begin(), v.begin(), w.begin()),
        SHELL_DIST_POW())));
}
