#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <bphcuda/cell_indexing.h>

#include <gtest/gtest.h>

TEST(CellIndexing, Test){
  size_t n_particle = 7;
  size_t _input[] = {0,0,0,1,1,3,4}; vector<size_t>::type input(_input, _input+n_particle);
  size_t n_cell = 5;
  vector<size_t>::type prefix(n_cell);
  vector<size_t>::type size(n_cell);
  
  bphcuda::cell_indexing(n_particle, input.begin(), n_cell, prefix.begin(), size.begin());     

  size_t _ans_prefix[] = {0,3,5,5,6}; vector<size_t>::type ans_prefix(_ans_prefix, _ans_prefix+n_cell);
  size_t _ans_size[] = {3,2,0,1,1}; vector<size_t>::type ans_size(_ans_size, _ans_size+n_cell);

  EXPECT_EQ(thrusting::make_list(ans_prefix), thrusting::make_list(prefix));
  EXPECT_EQ(thrusting::make_list(ans_size), thrusting::make_list(size));
}
