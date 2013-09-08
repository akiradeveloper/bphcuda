#pragma once

#include <thrusting/real.h>

namespace thrusting {

__host__ __device__
real sinr(real rad){
#ifdef THRUSTING_USING_DOUBLE_FOR_REAL
  return sin(rad);
#else
  return sinf(rad);
#endif
}

__host__ __device__
real cosr(real rad){
#ifdef THRUSTING_USING_DOUBLE_FOR_REAL
  return cos(rad);
#else
  return cosf(rad);
#endif
}

__host__ __device__
real sqrtr(real x){
#ifdef THRUSTING_USING_DOUBlE_FOR_REAL
  return sqrt(x);
#else
  return sqrtf(x);
#endif
}

} // END thrusting
