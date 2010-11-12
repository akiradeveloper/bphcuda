#include <thrust/transform_reduce.h>

#include <thrusting/functional.h>
#include <thrusting/iterator/zip_iterator.h>

struct momentum_calculator :public thrust::unary_function<real4, real3>{

}

template<typename R1, typename R2>
real3 calc_momentum(
  size_t n_particle,
  R1 u, R1 v, R1 w, 
  R2 m
){
  thrust::transform_reduce(
    

}
