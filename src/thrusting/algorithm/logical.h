#pragma once

#include <thrusting/algorithm/advance.h>

#include <thrust/logical.h>

namespace thrusting {

template<
typename Size,
typename InputIterator,
typename Predicate>
bool all_of(
  Size n,
  InputIterator first,
  Predicate pred
){
  return thrust::all_of(
    first,
    thrusting::advance(n, first),
    pred);
}

template<
typename Size,
typename InputIterator,
typename Predicate>
bool any_of(
  Size n,
  InputIterator first,
  Predicate pred
){
  return thrust::any_of(
    first,
    thrusting::advance(n, first),
    pred);
}

template<
typename Size,
typename InputIterator,
typename Predicate>
bool none_of(
  Size n,
  InputIterator first,
  Predicate pred
){
  return thrust::none_of(
    first,
    thrusting::advance(n, first),
    pred);
}

} // END thrusting
