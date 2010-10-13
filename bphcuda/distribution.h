#pragma once

#include <bphcuda/real.h>

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/random.h>

namespace bphcuda {

struct shell_rand :public thrust::unary_function<thrust::tuple<Real, Real>, Real3> {
  __host__ __device__
  Real3 operator()(thrust::tuple<Real, Real> rand){
    Real a = 2 * PI * rand.get<0>();
    Real b = 2 * PI * rand.get<1>();
    Real cx = cosf(a) * cosf(b);
    Real cy = cosf(a) * sinf(b);
    Real cz = sinf(a);
    return mk_real3(cx, cy, cz);
  }
};

struct shell_rand_adapter :public thrust::unary_function<Int, Real3> {
  Int seed;
  randf(Int seed_)
  :seed(seed_){}
  __device__ __host__
  float operator()(Int ind){
    thrust::default_random_engine rng(seed);
    rng.discard(2 * ind);
    thrust::uniform_real_distribution<Real> a01(0,1);
    thrust::uniform_real_distribution<Real> b01(0,1);
    return shell_rand()(thrust::make_tuple(a01(rng), b01(rng))); 	 
  }
};

// first to last shared same seed 
template<typename Iter>
void alloc_shell_rand(Iter first, Iter last, Int seed){
  Int len = last - first;
  transform(
    thrust::constant_iterator<Int>(1),
    thrust::constant_iterator<Int>(len+1),
    first,
    shell_rand_adapter());
}

typedef thrust::tuple<Real, Real, Real, Real, Real, Real> Real6;
struct maxwell_rand :public thrust::unary_function<Real6, Real3> {
  Real T;
  Real m;
  rand_shell(Real T_, Real m_)
  :T(T_), m(m_){}
  __host__ __device__
  Real3 operator()(Real6 rand){
    Real A = __sqrt( -2 Kb * T / m);
    cx = A * __logf(rands.get<0>()) * __cosf(
  }
};

template<typename Iter>
void alloc_maxwell_rand(){
}

} // end of bph
