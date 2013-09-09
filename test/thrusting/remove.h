#pragma once

#include <thrusting/algorithm/remove.h>
#include <thrusting/functional.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;

}

struct REMOVE_IS_ODD :public thrust::unary_function<int, bool> {
  __host__ __device__
  bool operator()(int x) const {
    return (x % 2) == 1;
  }
};

TEST(SortOutIf, Test){
  int _xs[] = {1,0,1}; vector<int>::type xs(_xs, _xs+3);

  /*
    the output for removed elements
    can be on different space.
  */
  thrust::host_vector<int> out(3);
  
  size_t n = thrusting::sort_out_if(
    3,
    xs.begin(),
    out.begin(),
    REMOVE_IS_ODD());

  EXPECT_EQ(1, n);
  
  int _ans[] = {1,1};
  vector<int>::type ans(_ans, _ans+2);
  EXPECT_EQ(
    make_list(ans), 
    make_list(2, out.begin()));
}
