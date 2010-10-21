#include <bphcuda/parallel.h>

#include "util.h"

using namespace bphcuda;
int main(void){
  thrust::device_vector<int> xs;
  xs.push_back(1);
  thrust::device_vector<int> ys;
  ys.push_back(2);
  
  parallel(1, xs.begin()) + ys.begin();
  std::cout << xs << std::endl;
  return 0;
}
