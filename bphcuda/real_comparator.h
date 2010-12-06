#pragma once

#include <thrust/functional.h>

#include <thrusting/real.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {
namespace detail {

__host__ __device__
real REAL_COMPARATOR_ABS(real a, real b){
  return (a>b) ? (a-b) : (b-a);
}

class real_comparator :public thrust::binary_function<real, real, bool> {
  real _scale;
  real _err;
public:
  real_comparator(real scale, real err)
  :_scale(scale), _err(err){}
  __host__ __device__
  bool operator()(real a, real b) const {
    real relative_err = REAL_COMPARATOR_ABS(a, b) / _scale;
    return relative_err < _err;
  }
};
} // END detail

__host__ __device__ 
detail::real_comparator make_real_comparator(real scale, real err){
  return detail::real_comparator(scale, err);
}

namespace detail {
struct real3_comparator :public thrust::binary_function<real3, real3, bool> {
  real3 _scale;
  real3 _err;
public:
  real3_comparator(real3 scale, real3 err)
  :_scale(scale), _err(err){}
  __host__ __device__
  bool operator()(const real3 &a, const real3 &b) const {
    bool bx = make_real_comparator(_scale.get<0>(), _err.get<0>())(a.get<0>(), b.get<0>());
    bool by = make_real_comparator(_scale.get<1>(), _err.get<1>())(a.get<1>(), b.get<1>());
    bool bz = make_real_comparator(_scale.get<2>(), _err.get<2>())(a.get<2>(), b.get<2>());
    return bx && by && bz;
  }
};
} // END detail

__host__ __device__
detail::real3_comparator make_real3_comparator(real3 scale, real err){
  return detail::real3_comparator(scale, real3(err, err, err));
}

} // END bphcuda
