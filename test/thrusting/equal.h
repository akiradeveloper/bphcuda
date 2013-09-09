#pragma once

#include <thrusting/algorithm/equal.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <gtest/gtest.h>

struct EQUAL_COMPARATOR :public thrust::binary_function<int, int, bool> {
  __host__ __device__
  bool operator()(int x, int y) const {
    return (y-x) == 1;
  }
};

TEST(Equal, Test){
  EXPECT_TRUE(
    thrusting::equal(
      10,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(1),
      EQUAL_COMPARATOR()));
}
