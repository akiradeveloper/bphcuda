#pragma once

#include <thrust/sort.h>

#include <thrusting/algorithm/advance.h>

namespace thrusting {

/*
  simple wrapper
*/
template<
typename Size,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
void sort_by_key(
  Size n,
  RandomAccessIterator1 key,
  RandomAccessIterator2 value
){
  thrust::sort_by_key(
    key,
    thrusting::advance(n, key),
    value);
}

} // END thrusting
