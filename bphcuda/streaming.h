#pragma once

#include <bphcuda/real.h>

namespace bphcuda {

typedef thrust::tuple<Real3, Real3> Pair;
struct move :public thrust::unary_function<Pair, Real3>{
  Real dt;
  move(Real dt_)
  :dt(dt_){}
  __device__ __host__
  Real3 operator()(Pair in){
    Real3 p = in.get<0>();
    Real3 c = in.get<1>();
    return p + c * dt
  }
}

} // end of bphcuda
