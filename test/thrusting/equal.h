#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrusting/algorithm/equal.h>

#include <gtest/gtest.h>


TEST(Equal, Test){
  using namespace thrust::placeholders;
  int xs[5] = { 1, 2, 3, 4, 5 };
  // int ys[5] = { 2, 3, 4, 5, 6 };

  /*
   * equal(counting_iter, counting_iter)
   * fails type inference some how.
   */
  bool r = thrusting::equal(
    5,
    xs,
    thrust::make_counting_iterator(2),
    _2 - _1 == 1);

  EXPECT_TRUE(r);
}
