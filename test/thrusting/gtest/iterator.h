#pragma once

#include <thrusting/tuple.h>
#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/vectorspace.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Iterator, IteratorValueAt){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  EXPECT_EQ(1, thrusting::iterator_value_at(0, xs.begin()));
  EXPECT_EQ(2, thrusting::iterator_value_at(1, xs.begin()));
}

TEST(Iterator, IteratorValueOf){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  EXPECT_EQ(1, thrusting::detail::iterator_value_of(xs.begin()));
}

TEST(Iterator, Advance){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  int _ys[] = {3,4}; vector<int>::type ys(_ys, _ys+2);
  EXPECT_EQ(thrust::make_tuple(2,4), 
    thrusting::detail::iterator_value_of(
      thrusting::advance(1, thrusting::make_zip_iterator(xs.begin(), ys.begin()))));
}

TEST(Iterator, AllocAt){
  int _xs[] = {1,2}; vector<int>::type xs(_xs, _xs+2);
  int _ys[] = {3,4}; vector<int>::type ys(_ys, _ys+2);
  thrusting::alloc_at(1, thrusting::make_zip_iterator(xs.begin(), ys.begin()), thrust::make_tuple(5,6));
  int _ans_xs[] = {1,5}; vector<int>::type ans_xs(_ans_xs, _ans_xs+2); 
  int _ans_ys[] = {3,6}; vector<int>::type ans_ys(_ans_ys, _ans_ys+2); 
  EXPECT_EQ(thrusting::make_list(ans_xs), thrusting::make_list(xs));
  EXPECT_EQ(thrusting::make_list(ans_ys), thrusting::make_list(ys));
}
