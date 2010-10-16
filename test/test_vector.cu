#include "util.h"

#include <bphcuda/vector.h>

using namespace bphcuda;

int main(void){
  // device
  thrust::device_vector<int> d_xs;
  d_xs.push_back(10);
  // std::cout << d_xs << std::endl;

  // host
  thrust::host_vector<int> h_xs;
  h_xs.push_back(20);
  // std::cout << h_xs << std::endl;
  
  return 0;
}
