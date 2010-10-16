#include <bphcuda/tuple3.h>

#include "util.h"

using namespace bphcuda;

int main(void){
  thrust::tuple<Int, Real, Real> t1 = thrust::make_tuple(0, 1.0, 2.0);
  thrust::tuple<Int, Real, Real> t2 = thrust::make_tuple(0, 1.0, 2.0);

  // equality
  ASSERT_EQUAL(t1, t2);

  // stdio
  std::cout << t1 << std::endl;
  return 0;
}
