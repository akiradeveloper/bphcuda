#pragma once

#include <thrusting/iterator/stride_iterator.h>
#include <thrusting/list.h>
#include <thrusting/vector.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(StrideIterator, Test){
  size_t len = 4;
  int _ans[] = {1,3,5,7}; vector<int>::type ans(_ans, _ans+len);

  int first = 1;
  int step = 2;

  EXPECT_EQ(
    make_list(ans),
    make_list(len, make_stride_iterator(first, step)));
}
