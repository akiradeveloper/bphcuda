#include "util.h"

#include <bphcuda/relaxing.h>
#include <iostream>

using namespace bphcuda;

int main(void){
  Real3 x = mk_real3(10.0f, 20.0f, 30.0f);
  Real3 y = mk_real3(40.0f, 50.0f, 60.0f);
  Real3 ps[2] = {x, y};
  thrust::device_vector<Real3> d_ps(ps, ps+2);
  relax(d_ps.begin(), d_ps.end(), 0);
  return 0;
}
