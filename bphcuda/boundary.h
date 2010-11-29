#pragma once

#include <thrusting/functional.h>
#include <thrusting/real.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {
  
struct mirroring :public thrust::unary_function<real, real> {
  real _middle;
  mirroring(real middle)
  :_middle(middle){}
  real operator()(real x) const {
    return 2 * _middle - x;
  }
};

struct retrive_greater :public thrust::unary_function<real, real>{
  real2 _range;
  retrive_bigger(real2 range)
  :_range(range){}
  real operator()(real x) const {
    real lower = _range.get<0>();
    real upper = _range.get<1>();
    real len = upper - lower;
    size_t time = (x - lower) / len;
    return x - time * len;
  }
};

struct retrive_less :public thrust::unary_function<real, real>{
  real2 _range;
  retrive_bigger(real2 range)
  :_range(range){}
  real operator()(real x) const {
    real lower = _range.get<0>();
    real upper = _range.get<1>();
    real len = upper - lower;
    size_t time = (upper - x) / len;
    return x + time * len;
  }
};

} // END bphcuda
