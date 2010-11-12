#pragma once

#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <bphcuda/const_value.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real6;
}

namespace bphcuda {

__device__ __host__
real calc_A(real rand, real m, real T){
  return sqrtf(real(-2.0) * BOLTZMANN() * T / m * logf(rand));
}

__device__ __host__
real calc_B(real rand){
  return cosf(real(2.0) * PI() * rand);
}

__device__ __host__
real calc_maxwell(real rand1, real rand2, real m, real T){
  return calc_A(rand1, m, T) * calc_B(rand2);
}

/*
  6 * rand -> c
*/
struct maxwell_rand :public thrust::unary_function<real6, real3> {
  real _m; // The mass of the particle
  real _T; // The temperature of the system
  maxwell_rand(real m, real T)
  :_m(m), _T(T){}

  __host__ __device__
  real3 operator()(real6 rand) const {
    real cx = calc_maxwell(rand.get<0>(), rand.get<1>(), _m, _T);
    real cy = calc_maxwell(rand.get<2>(), rand.get<3>(), _m, _T);
    real cz = calc_maxwell(rand.get<4>(), rand.get<5>(), _m, _T);
    return real3(cx, cy, cz);
  }
};

/*
  (m, idx) -> c
*/
struct maxwell_rand_generator :public thrust::binary_function<real, size_t, real3> {
  real _T;
  size_t _seed;
  maxwell_rand_generator(real T, size_t seed)
  :_T(T), _seed(seed){}

  __host__ __device__
  real3 operator()(real m, size_t idx) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 6;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u01(0.0, 1.0);
    // m is unique to each particle
    return maxwell_rand(m, _T)(
      thrusting::make_tuple6<real>(
        u01(rng), u01(rng),
        u01(rng), u01(rng),
        u01(rng), u01(rng)));
  }
};

/*
  T is constant because the system is thermally balanced.
  but m is not.
  [Real3] -> [Real3]
*/
template<typename RealIterator, typename RealIterator2>
void alloc_maxwell_rand(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w, // output
  RealIterator2 m,
  real T, 
  size_t seed 
){
  thrust::transform(
    m,
    thrusting::advance(n_particle, m),
    thrust::make_counting_iterator<size_t>(0),
    thrusting::make_zip_iterator(u, v, w),
    maxwell_rand_generator(T, seed));
}

} // END bphcuda

