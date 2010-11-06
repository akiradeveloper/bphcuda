#include "util.h"

#include <bphcuda/relaxing.h>
#include <iostream>

using namespace bphcuda;
int main(void){
  Real3 a = mk_real3(1.0, 2.0, 3.0);
  Real3 b = mk_real3(4.0, 5.0, 6.0);
  Real3 c = mk_real3(7.0, 8.0, 9.0);
  Real3 ps[3] = {a, b, c};
  thrust::device_vector<Real3> d_ps(ps, ps+3);
  Real before_e = calc_kinetic_e(d_ps.begin(), d_ps.end());
  const Int seed = 0;
  relax(d_ps.begin(), d_ps.end(), seed);
  Real after_e = calc_kinetic_e(d_ps.begin(), d_ps.end());
  Real3 after_momentum = thrust::reduce(d_ps.begin(), d_ps.end(), mk_real3(0,0,0));

  ASSERT_EQUAL(d_ps[2], mk_real3(0,0,0));
  ASSERT_EQUAL(d_ps[0], d_ps[1]*(-1));
  ASSERT_NEARLY_EQUAL(before_e, after_e, 1.0);
  ASSERT_EQUAL(after_momentum, mk_real3(0,0,0));
  
  return 0;
}
