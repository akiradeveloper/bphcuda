#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <string>
#include <sstream>

namespace bphcuda {

template<typename Iter>
std::string _to_s(Iter first, Iter last){
  std::stringstream ss;
  ss << "[";
  int size = last - first;
  for(int i=0; i<size; i++){
    ss << *(first+i) << ",";
  }
  std::string s = ss.str();
  std::string stripped = s.substr(0, s.size()-1);
  return stripped + "]";
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const thrust::device_vector<T> &xs){
  os << _to_s(xs.begin(), xs.end());
  return os;
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const thrust::host_vector<T> &xs){
  os << _to_s(xs.begin(), xs.end());
  return os;
}
  
} // end of bphcuda
