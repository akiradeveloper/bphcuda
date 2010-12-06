#pragma once

#include <thrusting/functional.h>
#include <thrusting/real.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {
  
namespace detail {
struct mirroring :public thrust::unary_function<real, real> {
  real _middle;
  mirroring(real middle)
  :_middle(middle){}
  real operator()(real x) const {
    return 2 * _middle - x;
  }
};
} // END detail 

detail::mirroring make_mirroring_functor(real middle){
  return detail::mirroring(middle);
}

namespace detail {
struct retrieve_greater :public thrust::unary_function<real, real>{
  real2 _range;
  retrieve_greater(real2 range)
  :_range(range){}
  real operator()(real x) const {
    real lower = _range.get<0>();
    real upper = _range.get<1>();
    real len = upper - lower;
    size_t time = (x - lower) / len;
    return x - time * len;
  }
};
} // END detail 

detail::retrieve_greater make_retrieve_greater_functor(real min, real max){
  return detail::retrieve_greater(real2(min, max));
}

namespace detail {
struct retrieve_less :public thrust::unary_function<real, real>{
  real2 _range;
  retrieve_less(real2 range)
  :_range(range){}
  real operator()(real x) const {
    real lower = _range.get<0>();
    real upper = _range.get<1>();
    real len = upper - lower;
    size_t time = (upper - x) / len;
    return x + time * len;
  }
};
} // END detail

detail::retrieve_less make_retrieve_less_functor(real min, real max){
  return detail::retrieve_less(real2(min, max));
}

} // END bphcuda
