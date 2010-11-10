#pragma once

#include <thrust/random.h>

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>

namespace {
  using thrusting::real;
  using thrusting::real2;
  using thrusting::real3;
}

namespace bphcuda {

namespace {
struct uniform_rand_generator :public thrust::unary_function<size_t, real> {
  real2 _range;
  size_t _seed;
  shell_rand_adapter(real2 range, size_t seed)
  :_range(range), _seed(seed){}

  __host__ __device__
  real operator()(size_t idx) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 1;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u_lower_upper(range.get<0>(), range.get<1>());
    return u_lower_upper(rng); 	 
  }
};
} // END namespace 

// impl but not tested
/*
  generate uniform random number [lower, upper] to array
*/
template<typaname RealIterator>
alloc_uniform_random(
  size_t n_particles,
  RealIterator begin,
  thrusting::real2 range,
  size_t seed
){
  thrust::transform(
    thrust::counting_iterator<size_t>(1),
    thrust::counting_iterator<size_t>(n_particle + 1),
    begin,
    uniform_random_generator(range, seed));
}

} // END bphcuda
