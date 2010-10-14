#include "util.h"

#include <bphcuda/relaxing.h>
#include <iostream>

using namespace bphcuda;

int main(void){
  Real3 x = mk_real3(10.0, 20.0, 30.0);
  Real3 y = mk_real3(40.0, 50.0, 60.0);
  Real3 ps[2] = {x, y};
  thrust::device_vector<Real3> d_ps(ps, ps+2);
  const int seed = 0;
  relax(d_ps.begin(), d_ps.end(), seed);
  return 0;
}
