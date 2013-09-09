#pragma once

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/algorithm/transform.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Transform, TransformIf){
  size_t n = 3;
  bool _stencil[] = {false,true,true}; vector<bool>::type stencil(_stencil, _stencil+n);
  int _value[] = {1,2,3}; vector<int>::type value(_value, _value+3);

  vector<int>::type output(n);

  thrusting::transform_if(
    n,
    value.begin(),
    stencil.begin(),
    output.begin(),
    thrust::identity<int>(), // op
    thrust::identity<bool>()); // pred

  int _ans[] = {0,2,3}; vector<int>::type ans(_ans, _ans+n);

  EXPECT_EQ(
    make_list(ans),
    make_list(output));
}
