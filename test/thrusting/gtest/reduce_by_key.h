#pragma once

#include <thrust/reduce.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(ReduceByKey, Test){
  int _key[] = {1,1,1,3,3}; vector<int>::type key(_key, _key+5);
  int _value[] = {1,2,3,4,5}; vector<int>::type value(_value, _value+5);

  vector<int>::type key_output(2);
  vector<int>::type value_output(2);
  
  thrust::reduce_by_key(
    key.begin(),
    key.end(),
    value.begin(),
    key_output.begin(),
    value_output.begin());
    
  int _ans_key[] = {1,3}; vector<int>::type ans_key(_ans_key, _ans_key+2);
  int _ans_value[] = {6,9}; vector<int>::type ans_value(_ans_value, _ans_value+2); 

  EXPECT_EQ(
    make_list(ans_key),
    make_list(key_output));

  EXPECT_EQ(
    make_list(ans_value),
    make_list(value_output));
}
