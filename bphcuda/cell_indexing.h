#pragma once

#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

/*
  [Int] -> ([Int], [Int])
  sorted_int_array -> (prefix, count)
*/
namespace bphcuda {

/*
  Input must be sorted in advance
*/
template<typename IntIterator>
void cell_indexing(
  size_t n_particle,
  IntIterator cell_idx, // Which cell the particle belongs to
  size_t n_cell,
  IntIterator prefix, IntIterator count // output
){
} 

// deprecated
template<typename Input, typename Prefix, typename Size>
void cell_indexing(
  Input in_F, Input in_L,  
  Prefix prefix_F, Prefix prefix_L,
  Size size_F){
  thrust::counting_iterator<Int> search_begin(0);
  Int cell_size = prefix_L - prefix_F;
  thrust::lower_bound(in_F, in_L, search_begin, search_begin+cell_size, prefix_F);
  thrust::upper_bound(in_F, in_L, search_begin, search_begin+cell_size, size_F);
  thrust::transform(size_F, size_F+cell_size, prefix_F, size_F, thrust::minus<Int>());
}

} // end of bphcuda
