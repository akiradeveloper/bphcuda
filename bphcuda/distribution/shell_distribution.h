#pragma once

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

/*
  (rand, rand) -> c
*/
struct shell_rand :public thrust::unary_function<real2, real3> {
  real _PI;
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

struct shell_rand_generator :public thrust::unary_function<size_t, real3> {
  real _PI;
  size_t _seed;
  shell_rand_generator(size_t seed, real PI)
  :_seed(seed), _PI(PI){}
  __host__ __device__
  real3 operator()(size_t idx) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 2;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u01(0, 1);
    return shell_rand(_PI)(real2(u01(rng), u01(rng))); 	 
  }
};

template<typename RealIterator>
void alloc_shell_rand(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  size_t seed,
  real PI
){
  thrust::transform(
    thrust::make_counting_iterator<size_t>(0),
    thrusting::advance(n_particle, thrust::make_counting_iterator<size_t>(0)),
    thrusting::make_zip_iterator(u, v, w),
    shell_rand_generator(seed, PI)); 
}

// Future
//template<typename RealIterator>
//void alloc_shell_rand(
//  size_t n_particle,
//  RealIterator u, RealIterator v, RealIterator w,
//  RealIterator m,
//  real T, size_t seed 
//){
//}

} // end of bphcuda
