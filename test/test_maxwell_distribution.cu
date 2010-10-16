#include "util.h"

#include <bphcuda/maxwell_distribution.h>

using namespace bphcuda;

int main(void){
  Int count = 10;
  thrust::device_vector<Real3> output(count);
  Real T = 1;
  Real m = 1;
  alloc_maxwell_rand(output.begin(), output.end(), 0, T, m);
  std::cout << output << std::endl;

  return 0;
}
