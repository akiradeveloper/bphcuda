#pragma once

#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrusting/functional.h>
#include <thrusting/iterator.h>

namespace thrusting {

namespace detail {
template<typename A, typename B>
struct BUCKET_INDEXING_LOCAL_MINUS :public thrust::binary_function<A, B, A> {
  __host__ __device__
  A operator()(A a, B b) const {
    return a-b;
  }
};
} // END detail

/*
  Example,

  Input:
  [0,1,1,2,2,2] 
   
  Output:
  [0,1,3] for prefix
  [1,2,3] for cnt
*/
template<
typename Size1,
typename Size2,
typename InputIterator,
typename OutputIterator1,
typename OutputIterator2> 
void bucket_indexing(
  Size1 n_idx,
  InputIterator idx,
  Size2 n_bucket,
  OutputIterator1 prefix_bucket, OutputIterator2 cnt_bucket 
){
  typedef typename thrust::iterator_value<InputIterator>::type InputValue;
  /*
    should be search_begin(min of idx)
    but this function is restricted to idx >= 0
    for performance reason
  */
  thrust::counting_iterator<InputValue> search_begin(0); 

  thrust::lower_bound(
    idx,
    thrusting::advance(n_idx, idx),
    search_begin,
    thrusting::advance(n_bucket, search_begin),
    prefix_bucket);

  thrust::upper_bound(
    idx,
    thrusting::advance(n_idx, idx),
    search_begin,
    thrusting::advance(n_bucket, search_begin),
    cnt_bucket);

  typedef typename thrust::iterator_value<OutputIterator1>::type OutputValue1;
  typedef typename thrust::iterator_value<OutputIterator2>::type OutputValue2;
  thrust::transform(
    cnt_bucket,
    thrusting::advance(n_bucket, cnt_bucket),
    prefix_bucket,
    cnt_bucket,
    detail::BUCKET_INDEXING_LOCAL_MINUS<OutputValue2, OutputValue1>());
}

} // END thrusting
