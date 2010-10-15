#pragma once

#include <bphcuda/int.h>

#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace bphcuda {

template<typename Input, typename Prefix, typename Size>
__host__ __device__
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
