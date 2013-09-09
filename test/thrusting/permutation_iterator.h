#pragma once

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(PermutationIterator, Gather){
  int _map[] = {0,1,0,1}; vector<int>::type map(_map, _map+4);
  int _ans[] = {1,2,1,2}; vector<int>::type ans(_ans, _ans+4);

  EXPECT_EQ(
    4,
    make_list(ans).length());

  EXPECT_EQ(
    make_list(ans),
    make_list(
      4,
      thrust::make_permutation_iterator(
        thrust::make_counting_iterator(1), map.begin())));
}

TEST(PermutationIterator, Scatter){
  int _value[10] = {1,0,1,0,1,0,1,0,1,0};
  vector<int>::type value(_value, _value+10);
  
  int _idx[10] = {0,2,4,6,8,1,3,5,7,9};
  vector<int>::type idx(_idx, _idx+10);

  int _ans[10] = {1,0,0,1,1,0,0,1,1,0};
  vector<int>::type ans(_ans, _ans+10);

 
  vector<int>::type indices(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(10));

  thrust::sort_by_key( 
    idx.begin(), idx.end(),
    indices.begin());
 
  EXPECT_EQ(
    make_list(ans),
    make_list(
      10,
      thrust::make_permutation_iterator(
        value.begin(),
        indices.begin())));
}
