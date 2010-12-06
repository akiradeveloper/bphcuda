#pragma once

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/algorithm/copy.h>
#include <thrusting/random/engine.h>
#include <thrusting/random/distribution.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
/*
  (rand, rand) -> c
*/
class shell_rand :public thrust::unary_function<real2, real3> {
  real _PI;
public:
  shell_rand(real PI)
  :_PI(PI){}
  __host__ __device__
  real3 operator()(const real2 &rand) const {
    real a = 2 * _PI * rand.get<0>();
    real b = 2 * _PI * rand.get<1>();
    real cx = cosf(a) * cosf(b);
    real cy = cosf(a) * sinf(b);
    real cz = sinf(a);
    return real3(cx, cy, cz);
  }
};
} // END detail

template<typename Real, typename Int, typename Predicate>
void alloc_shell_rand(
  size_t n_particle,
  Real u, Real v, Real w,
  Int stencil,
  Predicate pred,
  size_t seed,
  real PI = 3.14
){
  thrusting::copy_if(
    n_particle,
    thrust::make_transform_iterator(
      thrusting::make_zip_iterator(
        thrust::make_transform_iterator(
          thrust::counting_iterator<size_t>(0),
          thrusting::compose(
            thrusting::make_uniform_real_distribution<real>(0,1),
            thrusting::make_fast_rng_generator(seed))),
        thrust::make_transform_iterator(
          thrust::counting_iterator<size_t>(n_particle),
          thrusting::compose(
            thrusting::make_uniform_real_distribution<real>(0,1),
            thrusting::make_fast_rng_generator(seed)))),
      detail::shell_rand(PI)),
    stencil,
    thrusting::make_zip_iterator(u, v, w),
    pred);       
}

template<typename Real>
void alloc_shell_rand(
  size_t n_particle,
  Real u, Real v, Real w,
  size_t seed,
  real PI = 3.14
){
  alloc_shell_rand(
    n_particle,
    u, v, w,
    thrust::make_constant_iterator(true),
    thrust::identity<bool>(),
    seed);
}

} // end of bphcuda
