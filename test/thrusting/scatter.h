#pragma once

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/algorithm/scatter.h>
#include <thrusting/iterator/zip_iterator.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Scatter, InPlaceForZipIterator){
  float _x[] = {1,2,3,4}; vector<float>::type x(_x, _x+4);
  long _y[] = {1,2,3,4}; vector<long>::type y(_y, _y+4);
  int _idx[] = {3,2,1,0}; vector<int>::type idx(_idx, _idx+4);
  
  thrusting::scatter(
    thrusting::make_zip_iterator(x.begin(), y.begin()),
    thrusting::make_zip_iterator(x.end(), y.end()),
    idx.begin(),
    thrusting::make_zip_iterator(x.begin(), y.begin()));
 
  float _ans_x[] = {4,3,2,1}; vector<float>::type ans_x(_ans_x, _ans_x+4);
  long _ans_y[] = {4,3,2,1}; vector<long>::type ans_y(_ans_y, _ans_y+4);

  EXPECT_EQ(make_list(ans_x), make_list(x));
  EXPECT_EQ(make_list(ans_y), make_list(y));
}

TEST(Scatter, InPlace){
  int _value[] = {1,2,3,4}; vector<int>::type value(_value, _value+4);
  int _idx[] = {3,2,1,0}; vector<int>::type idx(_idx, _idx+4);

  thrusting::scatter(
    value.begin(),
    value.end(),
    idx.begin(),
    value.begin()); 

  int _ans[] = {4,3,2,1}; vector<int>::type ans(_ans, _ans+4);
  
  EXPECT_EQ(
    make_list(ans),
    make_list(value));
}

TEST(Scatter, Test){
  int _value[10] = {1,0,1,0,1,0,1,0,1,0};
  vector<int>::type value(_value, _value+10);
  
  int _idx[10] = {0,2,4,6,8,1,3,5,7,9};
  vector<int>::type idx(_idx, _idx+10);

  vector<int>::type result(10);
  
  thrusting::scatter(
    value.begin(),
    value.end(),
    idx.begin(),
    result.begin());

  int _ans[10] = {1,0,0,1,1,0,0,1,1,0};
  vector<int>::type ans(_ans, _ans+10);
  
  EXPECT_EQ(
    make_list(ans),
    make_list(result));
}
