#pragma once

#include <bphcuda/macro.h>
#include <thrust/tuple.h>
#include <string>
#include <sstream>
#include <iostream>

namespace bphcuda {

// Akira Hayakawa noted 2010 10/15 9:23
// Tuple can not access the element by runtime index. It must be constant!
//__host__ __device__
//template<typename Tuple>
//int size(Tuple t){
//  return thrust::tuple_size< Tuple >::value;
//}

template<typename Tuple3>
__host__  __device__
bool are_equal(Tuple3 a, Tuple3 b){
  if ( a.get<0>() != b.get<0>() ) { return false; }
  if ( thrust::get<0>(a) != thrust::get<0>(b) ) { return false; }
  if ( thrust::get<1>(a) != thrust::get<1>(b) ) { return false; }
  if ( thrust::get<2>(a) != thrust::get<2>(b) ) { return false; }
  return true;
}

template<typename Tuple3>
__host__  __device__
bool operator==(Tuple3 a, Tuple3 b){
  return are_equal(a, b);
}

//template<typename Tuple3>
//__host__
//std::ostream& operator<<(std::ostream &os, Tuple3 t){
//  std::stringstream ss;
//  ss << "(";
//  ss << thrust::get<0>(t) << std::endl;
//  ss << thrust::get<1>(t) << std::endl;
//  ss << thrust::get<2>(t) << std::endl;
//  ss << ")";
//  os << ss.str();
//  return os;
//}

} // end of bphcuda
