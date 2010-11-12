#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <bphcuda/cell_indexing.h>

#include <gtest/gtest.h>

TEST(cell_indexing, test1){
  size_t n_particle = 7;
  size_t _input[] = {0,0,0,1,1,3,4}; THRUSTING_VECTOR<size_t> input(_input, _input+n_particle);
  size_t n_cell = 5;
  THRUSTING_VECTOR<size_t> prefix(n_cell);
  THRUSTING_VECTOR<size_t> size(n_cell);
  
  bphcuda::cell_indexing(n_particle, input.begin(), n_cell, prefix.begin(), size.begin());     

  size_t _ans_prefix[] = {0,3,5,5,6}; THRUSTING_VECTOR<size_t> ans_prefix(_ans_prefix, _ans_prefix+n_cell);
  size_t _ans_size[] = {3,2,0,1,1}; THRUSTING_VECTOR<size_t> ans_size(_ans_size, _ans_size+n_cell);
  EXPECT_EQ(thrusting::make_list(ans_prefix), thrusting::make_list(prefix));
  EXPECT_EQ(thrusting::make_list(ans_size), thrusting::make_list(size));
}
