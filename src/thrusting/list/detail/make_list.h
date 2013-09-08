#pragma once

#include <thrusting/list.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/distance.h>
#include <sstream>

namespace thrusting {

template<typename Iterator>
detail::list<typename ::thrust::iterator_value<Iterator>::type> make_list(Iterator first, Iterator last){
  return make_list(thrust::distance(first, last), first);
}

template<typename T>
detail::list<T> make_list(const thrust::host_vector<T> &xs){
  return make_list(xs.size(), xs.begin());
}

template<typename T>
detail::list<T> make_list(const thrust::device_vector<T> &xs){
  return make_list(xs.size(), xs.begin());
}

template<typename T>
detail::list<T> make_list(const std::vector<T> &xs){
  thrust::host_vector<T> h_xs(xs.begin(), xs.end());
  return make_list(h_xs.size(), h_xs.begin());
}

} // end thrusting
