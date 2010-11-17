#pragma once

#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

#include <bphcuda/cell.h>

namespace {
  using thrusting::real;
  using thrusting::real2;
  using thrusting::real6;
}

namespace bphcuda {
  
struct mirroring :public thrust::unary_function<real, real> {
  real _middle;
  mirroring(real middle)
  :_middle(middle){}
  real operator()(real x) const {
  }
};

struct retrive_greater :public thrust::unary_function<real, real>{
  real2 _range;
  retrive_bigger(real2 range)
  :_range(range){}
  real operator()(real x) const {
  }
};

struct retrive_less :public thrust::unary_function<real, real>{
  real2 _range;
  retrive_bigger(real2 range)
  :_range(range){}
  real operator()(real x) const {
  }
};

} // END bphcuda
