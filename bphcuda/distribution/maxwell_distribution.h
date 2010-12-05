#pragma once

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/random/engine.h>
#include <thrusting/random/distribution.h>
#include <thrusting/algorithm/copy.h>

#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

namespace detail {
__device__ __host__
real calc_A(real rand, real m, real T, real BOLTZMANN){
  return sqrtf(real(-2.0) * BOLTZMANN * T / m * logf(rand));
}

__device__ __host__
real calc_B(real rand, real PI){
  return cosf(real(2.0) * PI * rand);
}

__device__ __host__
real calc_maxwell(real rand1, real rand2, real m, real T, real BOLTZMANN, real PI){
  return calc_A(rand1, m, T, BOLTZMANN) * calc_B(rand2, PI);
}

/*
  6 * rand -> c
*/
struct maxwell_rand :public thrust::unary_function<real6, real3> {
  real _m; // The mass of the particle
  real _T; // The temperature of the system
  real _BOLTZMANN;
  real _PI;
  maxwell_rand(real m, real T, real BOLTZMANN, real PI)
  :_m(m), _T(T), _BOLTZMANN(BOLTZMANN), _PI(PI){}

  __host__ __device__
  real3 operator()(const real6 &rand) const {
    real cx = calc_maxwell(rand.get<0>(), rand.get<1>(), _m, _T, _BOLTZMANN, _PI);
    real cy = calc_maxwell(rand.get<2>(), rand.get<3>(), _m, _T, _BOLTZMANN, _PI);
    real cz = calc_maxwell(rand.get<4>(), rand.get<5>(), _m, _T, _BOLTZMANN, _PI);
    return real3(cx, cy, cz);
  }
};

/*
  (m, idx) -> c
*/
struct maxwell_rand_generator :public thrust::binary_function<real, size_t, real3> {
  real _T;
  size_t _seed;
  real _BOLTZMANN;
  real _PI;
  maxwell_rand_generator(real T, size_t seed, real BOLTZMANN, real PI)
  :_T(T), _seed(seed), _BOLTZMANN(BOLTZMANN), _PI(PI){}

  __host__ __device__
  real3 operator()(real m, size_t idx) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 6;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u01(0.0, 1.0);
    // m is unique to each particle
    return maxwell_rand(m, _T, _BOLTZMANN, _PI)(
      thrusting::make_tuple6<real>(
        u01(rng), u01(rng),
        u01(rng), u01(rng),
        u01(rng), u01(rng)));
  }
};
} // END detail

/*
  T is constant because the system is thermally balanced.
  but m is not.
  [Real3] -> [Real3]
*/
template<typename Real1>
void alloc_maxwell_rand(
  size_t n_particle,
  Real1 u, Real1 v, Real1 w, // output
  real m,
  real T, 
  size_t seed, 
  real BOLTZMANN = 1.38e-23,
  real PI = 3.14
){
//  thrust::transform(
//    m,
//    thrusting::advance(n_particle, m),
//    thrust::make_counting_iterator<size_t>(0),
//    thrusting::make_zip_iterator(u, v, w),
//    detail::maxwell_rand_generator(T, seed, BOLTZMANN, PI));
   thrusting::copy(
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
             thrusting::make_fast_rng_generator(seed))),
         thrust::make_transform_iterator(
           thrust::counting_iterator<size_t>(2 * n_particle),
           thrusting::compose(
             thrusting::make_uniform_real_distribution<real>(0,1),
             thrusting::make_fast_rng_generator(seed))),
         thrust::make_transform_iterator(
           thrust::counting_iterator<size_t>(3 * n_particle),
           thrusting::compose(
             thrusting::make_uniform_real_distribution<real>(0,1),
             thrusting::make_fast_rng_generator(seed))),
         thrust::make_transform_iterator(
           thrust::counting_iterator<size_t>(4 * n_particle),
           thrusting::compose(
             thrusting::make_uniform_real_distribution<real>(0,1),
             thrusting::make_fast_rng_generator(seed))),
         thrust::make_transform_iterator(
           thrust::counting_iterator<size_t>(5 * n_particle),
           thrusting::compose(
             thrusting::make_uniform_real_distribution<real>(0,1),
             thrusting::make_fast_rng_generator(seed)))),
       detail::maxwell_rand(m, T, BOLTZMANN, PI)),
     thrusting::make_zip_iterator(u, v, w));       
}

} // END bphcuda
