#pragma once

#include <bphcuda/real.h>

namespace bphcuda {

struct move {
  __device__ __host__
  Real3 operator()(Real3 p, Real3 vel, Real dt){
    return p + vel * dt
  }
}

} // end of bphcuda
