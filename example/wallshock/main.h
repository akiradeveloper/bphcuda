#pragma once

#include <thrust/sort.h>

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/real.h>
#include <thrusting/functional.h>

#include <thrusting/algorithm/transform.h>
#include <thrusting/algorithm/reduce_by_bucket.h>

#include <bphcuda/cell.h>
#include <bphcuda/bph.h>
#include <bphcuda/boundary.h>
#include <bphcuda/streaming.h>
#include <bphcuda/force.h>

#include <cstdlib>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

int main(int narg, char **args){
  size_t n_particle = atoi(args[1]);
  size_t n_cell = atoi(args[2]);

  real m = 1;
  thrust::constant_iterator<real> m_it(m);
  real s = 2;

  size_t n_particle_per_cell = n_particle / n_cell;
  cell c(real3(0,0,0), real3(real(1)/n_cell, 1, 1), tuple3<size_t>::type(n_cell, 1, 1));

  vector<real>::type x(n_particle);
  vector<real>::type y(n_particle);
  vector<real>::type z(n_particle);
  
  for(int i=0; i<n_cell; ++i){
    // alloc random location by cell
  }

  /*
    init thermal velocities are 0,
    also the in_e is 0
  */
  vector<real>::type u(n_particle);
  vector<real>::type v(n_particle);
  vector<real>::type w(n_particle);
  vector<real>::type in_e(n_particle);

  vector<size_t>::type idx(n_particle);

  vector<real>::type tmp1(n_cell);
  vector<real>::type tmp2(n_cell);
  vector<real>::type tmp3(n_cell);
  vector<real>::type tmp4(n_cell);
  vector<real>::type tmp5(n_cell);
  vector<real>::type tmp6(n_cell);
  vector<real>::type tmp7(n_cell);

  vector<size_t>::type tmp8(n_cell);
  vector<size_t>::type tmp9(n_cell);

  /*
    add velocity of -1 toward the wall at x=0
  */
  thrust::fill(
    u.begin(),
    u.end(),
    real(-1));

  size_t step = 1000;
  real dt = real(1) / step;
  for(int i=0; i<step; ++i){
    std::cout << "time: " << dt * i << std::endl;

    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));
   
    thrust::sort_by_key(
      idx.begin(), idx.end(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));
    /*
      measure the macro scopics
    */
    // reduce_by_bucket();
  
    /*
      processed by BPH routine
    */
    bph(
      n_particle,
      x.begin(), y.begin(), z.begin(),
      u.begin(), v.begin(), w.begin(),
      m,
      in_e.begin(),
      s,
      idx.begin(),
      n_cell,
      // real tmp
      tmp1.begin(), tmp2.begin(), tmp3.begin(), tmp4.begin(),
      tmp5.begin(), tmp6.begin(), tmp7.begin(),
      // int tmp
      tmp8.begin(), tmp9.begin(),
      i); // seed 
  
    /*
      Move
    */
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin(), m_it), // input
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin()), // output 
      make_runge_kutta_1_functor(
        make_no_force_generator(),
        dt));

    /*
      Boundary treatment
    */
    thrusting::transform_if(
      n_particle,
      y.begin(),
      y.begin(), // output
      y.begin(), // stencil
      make_retrieve_less_functor(0, 1),
      thrusting::bind2nd(
        thrust::less<real>(), real(0)));
  
    // not enough boudary treatment implementd
   
    
    /*
      if x < 0 then u -= u
    */
    thrusting::transform_if(
      n_particle,
      u.begin(), // input
      u.begin(), // output,
      x.begin(), // stencil,
      thrust::negate<real>(),
      thrusting::bind2nd(
        thrust::less<real>(),
        real(0)));

    /*
      if x < 0 then x -= x
    */
    thrusting::transform_if(
      n_particle,
      x.begin(), // input
      x.begin(), // output
      x.begin(), // stencil
      make_mirroring_functor(0),
      thrusting::bind2nd(
        thrust::less<real>(),
        real(0)));

  } // END for 
}
