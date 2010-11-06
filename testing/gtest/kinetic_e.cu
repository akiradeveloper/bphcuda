#include "util.h"

#include <bphcuda/kinetic_e.h>

using namespace bphcuda;

int main(void){
  thrust::device_vector<Real4> vec;
  vec.push_back(mk_real4(1,2,3,1));
  ASSERT_EQUAL(vec.size(), 1);

  Real e = calc_kinetic_e(vec.begin(), vec.end());
  ASSERT_EQUAL(e, 7.0);

  return 0;
}
