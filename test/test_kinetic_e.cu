#include "util.h"

#include <bphcuda/kinetic_e.h>

using namespace bphcuda;

int main(void){
  Real3 r = mk_real3(1.0, 2.0, 3.0);
  Real result = kinetic_e()(r); 	
  ASSERT_EQUAL(result, 14.0);  
  
  thrust::device_vector<Real3> vec;
  vec.push_back(r);
  ASSERT_EQUAL(vec.size(), 1);
  
  Real e = 0.0F;
  e = calc_kinetic_e(vec.begin(), vec.end());
  ASSERT_EQUAL(e, 14.0F);

  return 0;
}
