#pragma once

/*
  in-placeable gather algorithm

  Akira Hayakawa, 2010.
*/

/*
  DO NOT USE!!!
  often causes
  unspecified launch failure.
*/

#include <thrusting/iterator.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrusting/vector.h>

namespace thrusting {

/*
  simple wrapper. not tested
*/
template<
typename Size,
typename InputIterator1,
typename InputIterator2,
typename RandomAccessIterator,
typename OutputIterator,
typename Predicate>
void gather_if(
  Size n,
  InputIterator1 idx,
  InputIterator2 stencil,
  RandomAccessIterator input,
  OutputIterator result,
  Predicate pred
){
  thrust::gather_if(
    idx,
    thrusting::advance(n, idx),
    stencil,
    input,
    result,
    pred);
}
  
/*
  deprecated
*/
template<
typename InputIterator1,
typename InputIterator2,
typename OutputIterator>
void gather(
  InputIterator1 map_first,
  InputIterator1 map_last,
  InputIterator2 input,
  OutputIterator output
){
  if(input==output){
    // std::cout << "inplace" << std::endl;
    typename vector_of<InputIterator2>::type value_copy(
      input,
      thrusting::advance(map_last-map_first, input));
    thrust::gather(map_first, map_last, value_copy.begin(), output);
  } else {
    // std::cout << "default" << std::endl;
    thrust::gather(map_first, map_last, input, output);
  } 
}

} // END thrusting
