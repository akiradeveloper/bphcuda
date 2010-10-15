#pragma once

#include <thrust/binary_search.h>
#include <thrust/transform.h>

template<typename Input, typename Prefix, typename Size>
__host__ __device__
void cell_indexing(
  Input in_F, Input in_L,  
  Prefix prefix_F, Prefix prefix_L,
  Size size_F){
  thrust::lower_bound();
  thrust::upper_bound();
  thrust::transform();
}

