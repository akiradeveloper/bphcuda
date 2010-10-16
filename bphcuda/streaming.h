#pragma once

#include <bphcuda/real.h>

namespace bphcuda {

/*
Input [(xs, cs)]
*/
struct move :public thrust::unary_function<Real6, Real3>{
  Real dt;
  move(Real dt_)
  :dt(dt_){}
  __device__ __host__
  Real3 operator()(const Real6 &in){
    Real3 p = mk_real3(in.get<0>(), in.get<1>(), in.get<2>());
    Real3 c = mk_real3(in.get<3>(), in.get<4>(), in.get<5>());
    return p + c * dt
  }
};

} // end of bphcuda
