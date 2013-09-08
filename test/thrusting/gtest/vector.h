#pragma once

#include <thrusting/vector.h>

#include <gtest/gtest.h>


namespace {
  using namespace thrusting;
}

TEST(Vector, Device){
  vector_of<thrust::device_vector<float>::iterator >::type xs(10);
  xs.push_back(1.0);
  EXPECT_EQ(11, xs.size());
}

TEST(Vector, Host){
  vector_of<thrust::host_vector<float>::iterator >::type xs(10);
  EXPECT_EQ(10, xs.size());
}
