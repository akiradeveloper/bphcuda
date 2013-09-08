#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>

#include <thrusting/iterator.h>

#include <thrusting/random/engine.h>
#include <thrusting/random/distribution.h>

namespace thrusting {
/*
  RandomGenerator is Idx -> ValueType
*/
template<
typename OutputIterator,
typename RandomGenerator>
void generate(
  OutputIterator first,
  OutputIterator last,
  RandomGenerator gen
){
  thrust::copy( 
    thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0), gen),
    thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(last-first), gen),
    first);
}
} // END thrusting
