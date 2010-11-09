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

namespace bphcuda {

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

struct maxwell_rand :public thrust::unary_function<real6, real3> {
  real T; // The temperature of the system
  real m; // The mass of the particle
  maxwell_rand(real T_, real m_)
  :T(T_), m(m_){}
  __host__ __device__
  real3 operator()(real6 rand){
    real cx = calc_maxwell(rand.get<0>(), rand.get<1>(), T, m);
    real cy = calc_maxwell(rand.get<2>(), rand.get<3>(), T, m);
    real cz = calc_maxwell(rand.get<4>(), rand.get<5>(), T, m);
    return make_real3(cx, cy, cz);
  }
};

struct maxwell_rand_adapter :public thrust::unary_function<size_t, real3> {
  size_t seed;
  real T;
  real m;
  maxwell_rand_adapter(size_t seed_, real T_, real m_)
  :seed(seed_), T(T_), m(m_){}
  __host__ __device__
  real3 operator()(size_t ind){
    thrust::default_random_engine rng(seed);
    const size_t skip = 6;
    rng.discard(skip * ind);
    thrust::uniform_real_distribution<real> u01(0,1);
    return maxwell_rand(T, m)(
      thrusting::make_tuple<real>(
        u01(rng), u01(rng),
        u01(rng), u01(rng),
        u01(rng), u01(rng)));
  }
};

/*
  to be modified
  T is constant because the system is thermally balanced.
  but m
  [Real3] -> [Real3]
*/
template<typename RealIterator>
void alloc_maxwell_rand(
  size_t n_particle,
  RealIterator u, RealIterator v, RealIterator w,
  RealIteartor m,
  real T, size_t seed 
){
  // implementation is wrong now
  const Int len = cs_L - cs_F;  
  thrust::transform(
    thrust::counting_iterator<Int>(1),
    thrust::counting_iterator<Int>(len+1),
    cs_F,
    maxwell_rand_adapter(seed, T, m));
}

} // end of bphcuda

