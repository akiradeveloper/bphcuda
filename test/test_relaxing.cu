#include "util.h"

#include <bphcuda/relaxing.h>
#include <iostream>

using namespace bpucuda;

int main(void){
  Real3 x = mk_real3(10.0f, 20.0f, 30.0f);
  Real3 y = mk_real3(40.0f, 50.0f, 60.0f);
  thrust::device_vector<Real3> d_ps(2);
  d_ps.push_back(x);
  d_ps.push_back(y);
  return 0;

}
