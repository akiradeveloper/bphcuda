#pragma once

#include <thrust/iterator/counting_iterator.h>

#include <thrusting/real.h>
#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/tuple.h>

#include <iostream>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(List, MakeString){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2); 
  EXPECT_EQ("[1, 2]", thrusting::detail::make_string(thrusting::make_list(xs)));
}

TEST(List, Ostream){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2); 
  std::cout << thrusting::make_list(xs) << std::endl;
}

TEST(List, Ostream2){
  real _xs[] = {1.2, 2.9}; vector<real>::type xs(_xs, _xs+2); 
  std::cout << thrusting::make_list(2, xs.begin()) << std::endl;
}

TEST(List, Equality){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2); 
 
  int _ys[] = {1,2}; vector<int>::type ys(_ys, _ys+2); 
  EXPECT_EQ(
    thrusting::make_list(xs),
    thrusting::make_list(ys));
 
  int _zs[] = {1,3}; vector<int>::type zs(_zs, _zs+2); 
  EXPECT_NE(
    thrusting::make_list(xs),
    thrusting::make_list(zs)); 
}

/*
  Equality between vector of different space
*/
TEST(List, Equality2){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  EXPECT_EQ(
    make_list(xs),
    make_list(2, thrust::make_counting_iterator(1)));
}
