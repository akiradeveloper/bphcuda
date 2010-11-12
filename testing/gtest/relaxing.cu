#include <iostream>

#include <thrusting/dtype/real.h>
#include <thrusting/vector.h>
#include <thrusting/iterator.h>

#include <bphcuda/relaxing.h>

#include <gtest/gtest.h>

namespace {
  using thrusting::real;
  using thrusting::real3;
}

// case 3 particles 
TEST(relaxing, n_particle_even){
  size_t n_particle = 3;
  real _xs[] = {1.0, 4.0, 7.0}; THRUSTING_VECTOR<real> xs(_xs, _xs+n_particle);
  real _ys[] = {2.0, 5.0, 8.0}; THRUSTING_VECTOR<real> ys(_ys, _ys+n_particle);
  real _xs[] = {3.0, 6.0, 9.0}; THRUSTING_VECTOR<real> zs(_zs, _zs+n_particle);

  real mass = 1.0;
 
  real old_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      xs.begin(),
      ys.begin(),
      zs.begin(),
      thrust::constant_iterator<real>(mass));
      
  real new_kinetic_e =
    bphcuda::calc_kinetic_e(
      n_particle,
      xs.begin(),
      ys.begin(),
      zs.begin(),
      thrust::constant_iterator<real>(mass));

  // the last element is zero speed
  EXPECT_EQ(
    real3(0.0, 0.0, 0.0),
    thrusting::iterator_value_at(2, thrusting::make_zip_iterator(xs.begin(), ys.begin(), zs.begin())));
  // preserving the momentum
  // preserving the energy
  EXPECT_EQ(old_kinetic_e, new_kinetic_e);
  
  Real3 a = mk_real3(1.0, 2.0, 3.0);
  Real3 b = mk_real3(4.0, 5.0, 6.0);
  Real3 c = mk_real3(7.0, 8.0, 9.0);
  Real3 ps[3] = {a, b, c};
  thrust::device_vector<Real3> d_ps(ps, ps+3);
  Real before_e = calc_kinetic_e(d_ps.begin(), d_ps.end());
  const size_t seed = 0;
  relax(d_ps.begin(), d_ps.end(), seed);
  Real after_e = calc_kinetic_e(d_ps.begin(), d_ps.end());
  Real3 after_momentum = thrust::reduce(d_ps.begin(), d_ps.end(), mk_real3(0,0,0));

  ASSERT_EQUAL(d_ps[2], mk_real3(0,0,0));
  ASSERT_EQUAL(d_ps[0], d_ps[1]*(-1));
  ASSERT_NEARLY_EQUAL(before_e, after_e, 1.0);
  ASSERT_EQUAL(after_momentum, mk_real3(0,0,0));
}

// case 2 particles
TEST(relaxing, n_particle_odd){
}
