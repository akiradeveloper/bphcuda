#pragma once

/*
  Helper functions for iterator objects.
 
  Akira Hayakawa 2010
*/

#include <thrusting/algorithm/advance.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/fill.h>

namespace thrusting {

namespace detail {

template<typename Iterator>
typename thrust::iterator_value<Iterator>::type iterator_value_of(Iterator it){
  return *(it);
}

} // END detail

template<typename Index, typename Iterator>
typename thrust::iterator_value<Iterator>::type iterator_value_at(Index n, Iterator it){
  return detail::iterator_value_of(thrusting::advance(n, it));
}

template<typename Index, typename Iterator>
void alloc_at(Index idx, Iterator it, const typename thrust::iterator_value<Iterator>::type &x){
  thrust::fill_n(
    thrusting::advance(idx, it),
    1,
    x);
}

} // END thrusting
