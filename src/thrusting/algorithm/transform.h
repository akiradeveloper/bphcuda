#pragma once

#include <thrust/transform.h>

#include <thrusting/iterator.h>

namespace thrusting {

/*
  simple wrapper
  not tested
*/
template<
typename Size,
typename InputIterator1,
typename InputIterator2,
typename ForwardIterator,
typename UnaryFunction,
typename Predicate>
ForwardIterator transform_if(
  Size n,
  InputIterator1 first,
  InputIterator2 stencil,
  ForwardIterator result,
  UnaryFunction op,
  Predicate pred
){
  return thrust::transform_if(
    first,
    thrusting::advance(n, first),
    stencil,
    result,
    op,
    pred);
}

/*
  simple wrapper
  not tested
*/
template<
typename Size,
typename InputIterator,
typename OutputIterator,
typename UnaryFunction>
OutputIterator transform(
  Size n,
  InputIterator first,
  OutputIterator result,
  UnaryFunction op
){
  return thrust::transform(
    first,
    thrusting::advance(n, first),
    result,
    op);
}

} // END thrusting
