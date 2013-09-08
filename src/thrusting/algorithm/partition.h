#pragma once

#include <thrusting/algorithm/advance.h>

#include <thrust/partition.h>

namespace thrusting {

/*
  simple wrapper
  not tested
*/
template<
typename Size,
typename ForwardIterator,
typename Predicate>
Size partition(
  Size n,
  ForwardIterator first,
  Predicate pred
){
  ForwardIterator end;
  end = thrust::partition(
    first,
    thrusting::advance(n, first),
    pred);
  
  return thrust::distance(first, end);
}

} // END thrusting
