#pragma once

#include <thrust/host_vector.h>

template<typename T>
thrust::host_vector<T> &make_host_vector(size_t n, ...){
  thrust::host_vector<T> xs;
  va_list values;
  va_start(values, n);
  for(int i=0; i<n; i++){
    xs.push_back(var_arg(values, T));
  }
  va_end(values);  
  return xs; 
}
