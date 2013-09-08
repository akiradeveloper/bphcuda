#pragma once

#include <thrust/transform.h>
#include <thrust/functional.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Thrust, TransformIf){
  int _data[10] = {-5, 0, 2, -3, 2, 4, 0, -1, 2, 8}; vector<int>::type data(_data, _data+10);
  int _stencil[10] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0}; vector<int>::type stencil(_stencil, _stencil+10);

  thrust::transform_if(
    data.begin(),
    data.end(),
    stencil.begin(),
    data.begin(),
    thrust::negate<int>(),
    thrust::identity<int>());
  
  int _ans[10] = {5, 0, -2, -3, -2, 4, 0, -1, -2, 8}; vector<int>::type ans(_ans, _ans+10);
  
  EXPECT_EQ( 
    make_list(ans),
    make_list(data));
}
