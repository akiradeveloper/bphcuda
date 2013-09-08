#pragma once

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/algorithm/reduce_by_bucket.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}


TEST(ReduceByBucket, Test){
  size_t n_value = 5;
  int _idx[] = {1,1,2,2,2}; vector<int>::type idx(_idx, _idx+n_value);
  int _value[] = {1,2,3,4,5}; vector<int>::type value(_value, _value+n_value);
  
  size_t n_bucket = 4; 
  vector<int>::type prefix_output(n_bucket);
  vector<int>::type cnt_output(n_bucket);
  vector<int>::type value_output(n_bucket);

  int null_value = 10000;

  thrusting::reduce_by_bucket(
    n_value,
    idx.begin(),
    value.begin(),
    n_bucket,
    prefix_output.begin(),
    cnt_output.begin(),
    value_output.begin(),
    null_value); 

  int _ans_prefix[] = {0,0,2,5}; vector<int>::type ans_prefix(_ans_prefix, _ans_prefix+n_bucket);
  int _ans_cnt[] = {0,2,3,0}; vector<int>::type ans_cnt(_ans_cnt, _ans_cnt+n_bucket);
  int _ans_value[] = {null_value,3,12,null_value}; vector<int>::type ans_value(_ans_value, _ans_value+n_bucket);

  EXPECT_EQ(
    make_list(ans_prefix),
    make_list(prefix_output));
  
  EXPECT_EQ(
    make_list(ans_cnt),
    make_list(cnt_output));

  EXPECT_EQ(
    make_list(ans_value),
    make_list(value_output));
}

TEST(ReduceByBucket, Test2){
  size_t n_value = 7;
  int _idx[] = {0,1,1,2,2,2,5}; vector<int>::type idx(_idx, _idx+n_value);
  int _value[] = {1,2,3,4,5,6,7}; vector<int>::type value(_value, _value+n_value);
  
  size_t n_bucket = 6; 
  vector<int>::type prefix_output(n_bucket);
  vector<int>::type cnt_output(n_bucket);
  vector<int>::type value_output(n_bucket);

  int null_value = 10000;

  thrusting::reduce_by_bucket(
    n_value,
    idx.begin(),
    value.begin(),
    n_bucket,
    prefix_output.begin(),
    cnt_output.begin(),
    value_output.begin(),
    null_value); 

  int _ans_prefix[] = {0,1,3,6,6,6}; vector<int>::type ans_prefix(_ans_prefix, _ans_prefix+n_bucket);
  int _ans_cnt[] = {1,2,3,0,0,1}; vector<int>::type ans_cnt(_ans_cnt, _ans_cnt+n_bucket);
  int _ans_value[] = {1,5,15,null_value,null_value,7}; vector<int>::type ans_value(_ans_value, _ans_value+n_bucket);

  EXPECT_EQ(
    make_list(ans_prefix),
    make_list(prefix_output));
  
  EXPECT_EQ(
    make_list(ans_cnt),
    make_list(cnt_output));

  EXPECT_EQ(
    make_list(ans_value),
    make_list(value_output));
}

TEST(ReduceByBucket, Test3){
  size_t n_value = 3;
  int _idx[] = {1,2,2}; vector<int>::type idx(_idx, _idx+n_value);
  int _value[] = {3,4,5}; vector<int>::type value(_value, _value+n_value);
  
  size_t n_bucket = 4; 
  vector<int>::type prefix_output(n_bucket);
  vector<int>::type cnt_output(n_bucket);
  vector<int>::type value_output(n_bucket);

  int null_value = 10000;

  thrusting::reduce_by_bucket(
    n_value,
    idx.begin(),
    thrust::make_transform_iterator(
      value.begin(),
      thrust::identity<int>()),
    n_bucket,
    prefix_output.begin(),
    cnt_output.begin(),
    value_output.begin(),
    null_value); 

  int _ans_prefix[] = {0,0,1,3}; vector<int>::type ans_prefix(_ans_prefix, _ans_prefix+n_bucket);
  int _ans_cnt[] = {0,1,2,0}; vector<int>::type ans_cnt(_ans_cnt, _ans_cnt+n_bucket);
  int _ans_value[] = {null_value,3,9,null_value}; vector<int>::type ans_value(_ans_value, _ans_value+n_bucket);

  EXPECT_EQ(
    make_list(ans_prefix),
    make_list(prefix_output));
  
  EXPECT_EQ(
    make_list(ans_cnt),
    make_list(cnt_output));

  EXPECT_EQ(
    make_list(ans_value),
    make_list(value_output));
}
