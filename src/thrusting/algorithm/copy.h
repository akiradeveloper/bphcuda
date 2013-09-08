#pragma once

#include <thrusting/iterator.h>

#include <thrust/copy.h>

namespace thrusting {

/*
  simple wrapper 
  not tested
*/
template<
typename Size,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate>
void copy_if(
  Size n,
  InputIterator1 from,
  InputIterator2 stencil,
  OutputIterator to,
  Predicate pred
){
  thrust::copy_if(
    from,
    thrusting::advance(n, from),
    stencil,
    to,
    pred);
}

/*
  simple wrapper 
  not tested
*/
template<
typename Size,
typename InputIterator,
typename OutputIterator>
void copy(Size n, InputIterator from, OutputIterator to){
  thrust::copy(
    from,
    thrusting::advance(n, from),
    to);
}

} // END thrusting
