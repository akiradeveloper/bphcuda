#pragma once

#include <thrust/functional.h>

#include <thrusting/real.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {
namespace detail {
class real_comparator :public thrust::binary_function<real, real, bool> {
  real _err;
public:
  real_comparator(err)
  :_err(err){}
  __host__ __device__
  bool operator()(real a, real b) const {
    real true_value = real(0.5) * (a+b);
    real absolute_err = absf(a-b);
    real relative_err = absolute_err / true_value;
    return relative_err < _err;   
  }
};
} // END detail

/*
  check two values are equals according to the relative err.
*/
detail::real_comparator make_real_comparator(real err){
  return real_comparator(err);
}

} // END bphcuda
