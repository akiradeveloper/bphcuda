#include <thrust/device_vector.h>


template<typename T>
thrust::device_vector<T> &make_device_vector(int n, ...){
  thrust::device_vector<T> xs;

  return xs; 
}

