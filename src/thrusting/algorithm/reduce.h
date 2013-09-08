#pragma once 

#include <thrusting/algorithm/advance.h>

#include <thrust/reduce.h>

namespace thrusting {

/*
  simple wrapper
  not tested
*/
template<
typename Size,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator1,
typename OutputIterator2>
Size reduce_by_key(
  Size n,
  InputIterator1 key,
  InputIterator2 value,
  OutputIterator1 key_out,
  OutputIterator2 value_out
){
  thrust::pair<OutputIterator1, OutputIterator2> end;
  end = thrust::reduce_by_key(
    key,
    thrusting::advance(n, key),
    value,
    key_out,
    value_out);

  return thrust::distance(key_out, end.first);
}

} // END thrusting
