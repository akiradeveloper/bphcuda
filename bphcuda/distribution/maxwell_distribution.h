#pragma once

#include <thrusting/tuple.h>
#include <thrusting/functional.h>
#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include <bphcuda/const_value.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real6;
}

namespace {
__device__ __host__
real calc_A(real rand, real T, real m){
  return sqrtf(-2 * BOLTZMANN() * T / m * logf(rand));
}

__device__ __host__
real calc_B(real rand){
  return cosf(2*PI() * rand);
}

__device__ __host__
real calc_maxwell(real rand1, real rand2, real T, real m){
  return calc_A(rand1, T, m) * calc_B(rand2);
}

/*
  6 * rand -> c
*/
struct maxwell_rand :public thrust::unary_function<real6, real3> {
  real _T; // The temperature of the system
  real _m; // The mass of the particle
  maxwell_rand(real T, real m)
  :_T(T), _m(m){}

  __host__ __device__
  real3 operator()(real6 rand) const {
    real cx = calc_maxwell(rand.get<0>(), rand.get<1>(), _T, _m);
    real cy = calc_maxwell(rand.get<2>(), rand.get<3>(), _T, _m);
    real cz = calc_maxwell(rand.get<4>(), rand.get<5>(), _T, _m);
    return thrusting::make_real3(cx, cy, cz);
  }
};

struct maxwell_rand_generator :public thrust::binary_functiona<real, size_t, real3> {
  size_t _seed;
  real _T;
  maxwell_rand_generator(size_t seed, real T)
  :_seed(seed), _T(T){}

  __host__ __device__
  real3 operator()(real m, size_t idx) const {
    thrust::default_random_engine rng(seed);
    const size_t skip = 6;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u01(0,1);
    // m is unique to particle
    return maxwell_rand(_T, m)(
      thrusting::make_tuple<real>(
        u01(rng), u01(rng),
        u01(rng), u01(rng),
        u01(rng), u01(rng)));
  }
};
} // END namespace

namespace bphcuda {

/*
  to be modified
  T is constant because the system is thermally balanced.
  but m
  [Real3] -> [Real3]
*/
template<typename RealIterator>
void alloc_maxwell_rand(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w, // output
  RealIteartor m,
  real T, size_t seed 
){
  // implementation is wrong now
  const Int len = cs_L - cs_F;  
  thrust::transform(
    thrust::counting_iterator<Int>(1),
    thrust::counting_iterator<Int>(len+1),
    cs_F,
    maxwell_rand_generator(seed, T, m));
}

} // END bphcuda
