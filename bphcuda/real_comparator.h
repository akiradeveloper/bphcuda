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
  real_comparator(real err)
  :_err(err){}
  __host__ __device__
  bool operator()(real a, real b) const {
    real true_value = real(0.5) * (a+b);
    real absolute_err = a-b ? a-b : b-a;
    real relative_err = absolute_err / true_value;
    return relative_err < _err;   
  }
};
} // END detail

/*
  check two values are equals according to the relative err.
*/
__host__ __device__ 
detail::real_comparator make_real_comparator(real err){
  return detail::real_comparator(err);
}

namespace detail {
struct real3_comparator :public thrust::binary_function<real3, real3, bool> {
  real _err;
public:
  real3_comparator(real err)
  :_err(err){}
  __host__ __device__
  bool operator()(const real3 &a, const real3 &b) const {
    bool bx = make_real_comparator(_err)(a.get<0>(), b.get<0>());
    bool by = make_real_comparator(_err)(a.get<1>(), b.get<1>());
    bool bz = make_real_comparator(_err)(a.get<2>(), b.get<2>());
    return bx && by && bz;
  }
};
} // END detail

detail::real3_comparator make_real3_comparator(real err){
  return detail::real3_comparator(err);
}

} // END bphcuda
