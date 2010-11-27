#pragma once

#include <thrust/random.h>

#include <thrusting/real.h>

namespace {
  using namespace thrusting;
}

namespace bphcuda {

struct uniform_random_generator :public thrust::unary_function<size_t, real> {
  real2 _range;
  size_t _seed;
  uniform_random_generator(real2 range, size_t seed)
  :_range(range), _seed(seed){}
  __host__ __device__
  real operator()(size_t idx) const {
    thrust::default_random_engine rng(_seed);
    const size_t skip = 1;
    rng.discard(skip * idx);
    thrust::uniform_real_distribution<real> u_lower_upper(_range.get<0>(), _range.get<1>());
    return u_lower_upper(rng); 	 
  }
};

/*
  generate uniform random number [lower, upper] to array
*/
template<typename Real>
void alloc_uniform_random(
  size_t n_particle,
  Real begin,
  thrusting::real2 range,
  size_t seed
){
  size_t origin = 0;
  thrust::transform(
    thrust::counting_iterator<size_t>(origin),
    thrust::counting_iterator<size_t>(origin + n_particle),
    begin,
    uniform_random_generator(range, seed));
}

} // END bphcuda
