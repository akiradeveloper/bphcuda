#pragma once

#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/vector.h>

#include <thrusting/tuple.h>
#include <thrusting/vectorspace.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(ZipIterator, Reference){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  int _ys[] = {3,4}; vector<int>::type ys(_ys, _ys+2);

  EXPECT_EQ(
    thrust::make_tuple(1,3), 
    thrusting::iterator_value_at(0, thrusting::make_zip_iterator(xs.begin(), ys.begin())));

  EXPECT_EQ(
    thrust::make_tuple(2,4), 
    thrusting::iterator_value_at(1, thrusting::make_zip_iterator(xs.begin(), ys.begin())));
}
