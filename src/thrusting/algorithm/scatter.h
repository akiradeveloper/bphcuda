#pragma once

/*
  in-placeable scatter algorithm

  Akira Hayakawa, 2010.
*/

/*
  DO NOT USE!!!
  often causes
  unspecified launch failure.
*/

#include <thrust/iterator/iterator_traits.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrusting/vector.h>

namespace thrusting {

/*
  deprecated
*/
template<
typename InputIterator1,
typename InputIterator2,
typename OutputIterator>
void scatter(
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 map,
  OutputIterator output
){
  if(first==output){
//    std::cout << "inplace" << std::endl;
    typename vector_of<InputIterator1>::type value_copy(first, last);
    thrust::scatter(value_copy.begin(), value_copy.end(), map, output);
  } else {
 //   std::cout << "default" << std::endl;
    thrust::scatter(first, last, map, output);
  }
}

} // END thrusting
