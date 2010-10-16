#include "util.h"

#include <bphcuda/shell_distribution.h>

using namespace bphcuda;

int main(void){
  Int count = 1000000;
  thrust::device_vector<Real3> output(count);
  alloc_shell_rand(output.begin(), output.end(), 0);
  // std::cout << output << std::endl;
  return 0;
}
