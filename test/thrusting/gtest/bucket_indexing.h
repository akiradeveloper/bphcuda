#pragma once

#include <thrusting/vector.h>
#include <thrusting/list.h>
#include <thrusting/algorithm/bucket_indexing.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(BucketIndexing, Test){
  size_t n_particle = 7;
  size_t _input[] = {0,0,0,1,1,3,4}; vector<size_t>::type input(_input, _input+n_particle);
  size_t n_cell = 5;
  vector<size_t>::type prefix(n_cell);
  vector<size_t>::type size(n_cell);
  
  thrusting::bucket_indexing(n_particle, input.begin(), n_cell, prefix.begin(), size.begin());     

  size_t _ans_prefix[] = {0,3,5,5,6}; vector<size_t>::type ans_prefix(_ans_prefix, _ans_prefix+n_cell);
  size_t _ans_size[] = {3,2,0,1,1}; vector<size_t>::type ans_size(_ans_size, _ans_size+n_cell);

  EXPECT_EQ(make_list(ans_prefix), make_list(prefix));
  EXPECT_EQ(make_list(ans_size), make_list(size));
}
