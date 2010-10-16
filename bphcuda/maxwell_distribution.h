#pragma once

#include <bphcuda/real.h>
#include <bphcuda/constant.h>

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

namespace bphcuda {

__device__ __host__
Real sqrtlogf(Real x){
  return sqrtf(logf(x));
}

__device__ __host__
Real calc_else(Real x, Real y){
  return sqrtlogf(x) * cosf(2*PI() * y);
}

struct maxwell_rand :public thrust::unary_function<Real6, Real3> {
  Real T;
  Real m;
  maxwell_rand(Real T_, Real m_)
  :T(T_), m(m_){}
  __host__ __device__
  Real3 operator()(Real6 rand){
    Real A = sqrtf( -2 * BOLTZMANN() * T / m);
    Real cx = A * calc_else(rand.get<0>(), rand.get<1>());
    Real cy = A * calc_else(rand.get<2>(), rand.get<3>());
    Real cz = A * calc_else(rand.get<4>(), rand.get<5>());
    return mk_real3(cx, cy, cz);
  }
};

struct maxwell_rand_adapter :public thrust::unary_function<Int, Real3> {
  Int seed;
  Real T;
  Real m;
  maxwell_rand_adapter(Int seed_, Real T_, Real m_)
  :seed(seed_), T(T_), m(m_){}
  __host__ __device__
  Real3 operator()(Int ind){
    thrust::default_random_engine rng(seed);
    const Int skip = 6;
    rng.discard(skip * ind);
    thrust::uniform_real_distribution<Real> u01(0,1);
    return maxwell_rand(T, m)(
      thrust::make_tuple(
        u01(rng), u01(rng),
        u01(rng), u01(rng),
        u01(rng), u01(rng)));
  }
};

// [Real3] -> [Real3]
template<typename Iter>
void alloc_maxwell_rand(
  Iter cs_F, Iter cs_L,
  Int seed, Real T, Real m){
  const Int len = cs_L - cs_F;  
  thrust::transform(
    thrust::counting_iterator<Int>(1),
    thrust::counting_iterator<Int>(len+1),
    cs_F,
    maxwell_rand_adapter(seed, T, m));
}

} // end of bphcuda

