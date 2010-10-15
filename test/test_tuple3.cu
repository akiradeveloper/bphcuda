#include <bphcuda/tuple3.h>
#include <thrust/tuple.h>

#include "util.h"

using namespace bphcuda;

int main(void){
  thrust::tuple<int, float, double> t1 = thrust::make_tuple(0, 1.0, 2.0);
  thrust::tuple<int, float, double> t2 = thrust::make_tuple(0, 1.0, 2.0);

  // equality
  ASSERT_EQUAL(t1, t2);

  // stdio
  // std::cout << t1 << std::endl;
  return 0;
}
