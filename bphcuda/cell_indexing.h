#pragma once

#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrusting/iterator.h>

namespace bphcuda {

/*
  [Int] -> ([Int], [Int])
  sorted_int_array -> (prefix, count)
*/
template<typename IntIterator>
void cell_indexing(
  size_t n_particle,
  IntIterator cell_idx, // Which cell the particle belongs to
  size_t n_cell,
  IntIterator prefix, IntIterator count // output
){
  thrust::counting_iterator<size_t> search_begin(0);
  thrust::lower_bound(
    cell_idx,
    thrusting::advance(n_particle, cell_idx),
    search_begin,
    thrusting::advance(n_cell, search_begin),
    prefix);
  thrust::upper_bound(
    cell_idx,
    thrusting::advance(n_particle, cell_idx),
    search_begin,
    thrusting::advance(n_cell, search_begin),
    count);
  thrust::transform(
    count,
    thrusting::advance(n_cell, count),
    prefix,
    count,
    thrust::minus<size_t>());
}

} // end of bphcuda
