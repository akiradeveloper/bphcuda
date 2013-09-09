#pragma once

#include <thrusting/algorithm/copy.h>
#include <thrusting/list.h>
#include <thrusting/vector.h>

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Copy, Test){
  int n = 3;
  vector<int>::type dest(n);
  thrusting::copy(
    n,
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(1),
      thrust::negate<char>()),
    dest.begin());
  
  int _ans[] = {-1, -2, -3}; vector<int>::type ans(_ans, _ans+n);
  EXPECT_EQ(
    make_list(ans),
    make_list(dest));
}
