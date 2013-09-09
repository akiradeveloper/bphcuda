#pragma once

#include <thrust/sort.h>

#include <thrusting/real.h>
#include <thrusting/iterator/zip_iterator.h>
#include <thrusting/tuple.h>
#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Thrust, SortByKey){
  size_t n_data = 2;
  
  // TESTING of sort_by_key
  size_t _key[] = {2,1}; vector<size_t>::type key(_key, _key+n_data);
  size_t _value[] = {3,4}; vector<size_t>::type value(_value, _value+n_data);
  
  // sorting [(key, value)] by [key]
  thrust::sort_by_key(
    key.begin(), key.end(),
    // also testing zip1
    thrust::make_zip_iterator(thrust::make_tuple(value.begin())));

  size_t _ans_key[] = {1,2}; vector<size_t>::type ans_key(_ans_key, _ans_key+n_data);
  size_t _ans_value[] = {4,3}; vector<size_t>::type ans_value(_ans_value, _ans_value+n_data);

  // value is sorted
  EXPECT_EQ(thrusting::make_list(value), thrusting::make_list(ans_value));

  // key is also sorted
  EXPECT_EQ(thrusting::make_list(key), thrusting::make_list(ans_key));
}
