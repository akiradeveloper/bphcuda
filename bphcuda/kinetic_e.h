#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>
#include <thrusting/tuple.h>
#include <thrusting/iterator.h>
#include <thrusting/iterator/zip_iterator.h>

#include <thrust/transform_reduce.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real4;
}

namespace bphcuda {

__host__ __device__
real calc_kinetic_e(const real3 &c, real m){
  return real(0.5) * m * (
    c.get<0>() * c.get<0>() +
    c.get<1>() * c.get<1>() +
    c.get<2>() * c.get<2>());
}

/*
  (c, m) -> kinetic_e
*/
struct kinetic_e :public thrust::unary_function<real4, real> {
  __host__ __device__
  real operator()(const real4 &in) const {
    real3 c = real3(in.get<0>(), in.get<1>(), in.get<2>());
    real m = in.get<3>();
    return calc_kinetic_e(c, m);
  }
}; 

template<typename RealIterator>
real calc_kinetic_e(
  size_t n_particle, 
  RealIterator u, RealIterator v, RealIterator w, 
  RealIterator m
){
  return thrust::transform_reduce(
    thrusting::make_zip_iterator(u, v, w, m),
    thrusting::advance(n_particle, thrusting::make_zip_iterator(u, v, w, m)),
    kinetic_e(),
    real(0.0),
    thrust::plus<real>());
}

// Future
//template<typename RealIterator, typename IntIterator>
//real calc_kinetic_e(
//  size_t n_particle, 
//  RealIterator u, RealIterator v, RealIterator w, 
//  RealIterator m,
//  IntIterator n_group
//){
//}

} // END bphcuda
