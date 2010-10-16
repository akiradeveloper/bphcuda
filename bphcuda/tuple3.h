#pragma once

#include <bphcuda/macro.h>
#include <thrust/tuple.h>
#include <string>
#include <sstream>
#include <iostream>

namespace bphcuda {

template<typename A, typename B, typename C>
__host__  __device__
bool are_equal(const thrust::tuple<A,B,C> &a, const thrust::tuple<A,B,C> &b){
  if ( thrust::get<0>(a) != thrust::get<0>(b) ) { return false; }
  if ( thrust::get<1>(a) != thrust::get<1>(b) ) { return false; }
  if ( thrust::get<2>(a) != thrust::get<2>(b) ) { return false; }
  return true;
}

template<typename A, typename B, typename C>
__host__  __device__
bool operator==(const thrust::tuple<A,B,C> &a, const thrust::tuple<A,B,C> &b){
  return are_equal(a, b);
}

template<typename A, typename B, typename C>
__host__
std::ostream& operator<<(std::ostream &os, const thrust::tuple<A,B,C> &t){
  std::stringstream ss;
  ss << "(";
  ss << thrust::get<0>(t) << ", ";
  ss << thrust::get<1>(t) << ", ";
  ss << thrust::get<2>(t);
  ss << ")";
  os << ss.str();
  return os;
}

} // end of bphcuda
