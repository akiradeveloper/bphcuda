#pragma once

#include <thrusting/dtype/real.h>
#include <thrusting/dtype/tuple/real.h>
#include <thrusting/functional.h>
#include <thrusting/tuple.h>
#include <thrusting/iterator/zip_iterator.h>

#include <thrust/transform_reduce.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
  using thrusting::real4;
}

namespace {
// modified. not tested
/*
  c -> m -> kinetic_e
*/
struct kinetic_e :public thrust::unary_function<real4, real> {
  __host__ device__
  real operator()(const real4 &in) const {
    real3 c = thrusting::make_real3(in.get<0>(), in.get<1>(), in.get<2>());
    real m = in.get<3>();
    return bphcuda::calc_kinetic_e(c, m);
  }
}
} // END namespace

namespace bphcuda {

template<typename RealIterator>
real calc_kinetic_e(const real3 &c, real m){
  return 0.5 * m * (
    c.get<0>() * c.get<0>() +
    c.get<1>() * c.get<1>() +
    c.get<2>() * c.get<2>());
}
  
// impl but not tested
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
    0.0,
    thrust::plus<real>());
}

// Future
template<typename RealIterator, typename IntIterator>
real calc_kinetic_e(
  size_t n_particle, 
  RealIterator u, RealIterator v, RealIterator w, 
  RealIterator m,
  IntIterator n_group
){
}

// deprecated
// (c, m) -> e
struct kinetic_e :public thrust::unary_function<Real4, Real> {
  __host__ __device__
  Real operator()(const Real4 &x){
    Real3 v = mk_real3(x.get<0>(), x.get<1>(), x.get<2>());
    Real3 p = v * v;
    Real m = x.get<3>();
    return 0.5 * m * (p.get<0>() + p.get<1>() + p.get<2>());
  }
};

// deprecated
// [(c,m)] -> e
template<typename Particle>
Real calc_kinetic_e(Particle F, Particle L){
  return thrust::transform_reduce(F, L, kinetic_e(), 0.0F, thrust::plus<Real>());
}

} // end of bphcuda
