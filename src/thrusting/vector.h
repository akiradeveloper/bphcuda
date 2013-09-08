#pragma once

/*
  You can switch vector type by adding 

  -D THRUSTING_USING_DEVICE_VECTOR

  This is useful for building a portable software where
  you can benchmark the performance of your software between GPU and CPU
  without modified here and there in your software.
  
  Use 
  
  thrusting::vector<datatype>::type 

  instead writing thrust::device_vector or thrust::host_vector
  then you will be more easy to investigate GPU/CPU performance acceleration!
*/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace thrusting {

template<typename T>
struct vector {
#ifdef THRUSTING_USING_DEVICE_VECTOR
  typedef typename thrust::device_vector<T> type; 
#else
  typedef typename thrust::host_vector<T> type;
#endif
};

#include <thrust/detail/type_traits.h>

/*
  deprecated. 

  Generate either thrust::host_vector or thrust::device_vector in compilation
  according to the space of given Iterator.
*/
template<typename Iterator>
struct vector_of
:public thrust::detail::eval_if<
  thrust::detail::is_convertible<typename thrust::iterator_space<Iterator>::type, thrust::host_space_tag>::value,
  thrust::detail::identity_<thrust::host_vector<typename thrust::iterator_value<Iterator>::type> >,
  thrust::detail::identity_<thrust::device_vector<typename thrust::iterator_value<Iterator>::type> >
> 
{};

} // END thrusting
