#pragma once

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <thrusting/dtype/real.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>

#include <bphcuda/const_value.h>

namespace {
  using thrusting::real;
  using thrusting::real2;
  using thrusting::real3;
}

namespace bphcuda {

// (rand, rand) -> c
struct shell_rand :public thrust::unary_function<real2, real3> {
  __host__ __device__
  real3 operator()(const real2 &rand) const {
    real a = 2 * PI() * rand.get<0>();
    real b = 2 * PI() * rand.get<1>();
    real cx = cosf(a) * cosf(b);
    real cy = cosf(a) * sinf(b);
    real cz = sinf(a);
    return mk_real3(cx, cy, cz);
  }
};

struct shell_rand_adapter :public thrust::unary_function<size_t, real3> {
  size_t _seed;
  shell_rand_adapter(size_t seed)
  :_seed(seed){}

  __host__ __device__
  real3 operator()(size_t ind) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 2;
    rng.discard(skip * ind);
    thrust::uniform_real_distribution<real> u01(0,1);
    return shell_rand()(thrusting::make_tuple<real>(u01(rng), u01(rng))); 	 
  }
};

// deprecated
template<typename Velocity>
void alloc_shell_rand(Velocity cs_F, Velocity cs_L, Int seed){
  const Int len = cs_L - cs_F;
  thrust::transform(
    thrust::counting_iterator<Int>(1),
    thrust::counting_iterator<Int>(len+1),
    cs_F,
    shell_rand_adapter(seed));
}

template<typename R>
void alloc_shell_rand(
  size_t n_particle,
  R u, R v, R w,
  size_t seed
){
}

// Future
template<typename RealIterator>
void alloc_shell_rand(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  RealIterator m,
  real T, size_t seed 
){
}


} // end of bphcuda
