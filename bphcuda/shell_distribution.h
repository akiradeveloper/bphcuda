#pragma once

#include <bphcuda/real.h>
#include <bphcuda/constant.h>

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace bphcuda {

// (rand, rand) -> c
struct shell_rand :public thrust::unary_function<Real2, Real3> {
  __host__ __device__
  Real3 operator()(const Real2 &rand){
    Real a = 2 * PI() * rand.get<0>();
    Real b = 2 * PI() * rand.get<1>();
    Real cx = cosf(a) * cosf(b);
    Real cy = cosf(a) * sinf(b);
    Real cz = sinf(a);
    return mk_real3(cx, cy, cz);
  }
};

struct shell_rand_adapter :public thrust::unary_function<Int, Real3> {
  Int seed;
  shell_rand_adapter(Int seed_)
  :seed(seed_){}
  __host__ __device__
  Real3 operator()(Int ind){
    thrust::default_random_engine rng(seed);
    const Int skip = 2;
    rng.discard(skip * ind);
    thrust::uniform_real_distribution<Real> u01(0,1);
    return shell_rand()(thrust::make_tuple(u01(rng), u01(rng))); 	 
  }
};

template<typename Velocity>
void alloc_shell_rand(Velocity cs_F, Velocity cs_L, Int seed){
  const Int len = cs_L - cs_F;
  thrust::transform(
    thrust::counting_iterator<Int>(1),
    thrust::counting_iterator<Int>(len+1),
    cs_F,
    shell_rand_adapter(seed));
}

} // end of bphcuda