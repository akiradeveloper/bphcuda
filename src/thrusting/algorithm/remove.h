#pragma once

#include <thrusting/algorithm/advance.h>
#include <thrusting/algorithm/partition.h>
#include <thrusting/functional.h>

#include <thrust/distance.h>
#include <thrust/remove.h>

namespace thrusting {

/*
  simple wrapper, will not tested
*/
template<
typename Size,
typename InputIterator,
typename Predicate>
Size remove_if(
  Size n,
  InputIterator first,
  Predicate pred
){
  InputIterator end = thrust::remove_if(
    first,
    thrusting::advance(n, first),
    pred);

  return thrust::distance(first, end);
}

/*
  simple wrapper
*/
template<
typename Size,
typename InputIterator1,
typename InputIterator2,
typename Predicate>
Size remove_if(
  Size n,
  InputIterator1 first,
  InputIterator2 stencil,
  Predicate pred
){
  InputIterator1 end = thrust::remove_if(
    first,
    thrusting::advance(n, first),
    stencil,
    pred);

  return thrust::distance(first, end);
}

template<
typename Size,
typename InputIterator,
typename OutputIterator,
typename Predicate>
Size sort_out_if(
  Size n,
  InputIterator first,
  OutputIterator trashbox,
  Predicate pred
){
  Size n_remain = thrusting::partition(
    n,
    first,
    thrusting::compose(
      thrust::logical_not<bool>(),
      pred));        

  thrust::copy(
    thrusting::advance(n_remain, first),
    thrusting::advance(n, first),
    trashbox);

  return n_remain;
}

} // END thrusting
