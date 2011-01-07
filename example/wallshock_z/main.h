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
#include <bphcuda/random/uniform_random.h>

#include <cstdio>
#include <cstdlib>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

int main(int narg, char **args){
  char *filename = args[1];
  real s = atof(args[2]);
  size_t n = atoi(args[3]);
  
  /*
    parameter
  */
  size_t n_particle_per_cell = n;
  size_t n_cell = 1000;

  size_t n_particle = n_particle_per_cell * n_cell;

  real m = 1;
  thrust::constant_iterator<real> m_it(m);

  real z_origin = 1;
  cell c(real3(0,0,z_origin), real3(1, 1, real(1)/n_cell), tuple3<size_t>::type(1, 1, n_cell));

  vector<real>::type x(n_particle);
  vector<real>::type y(n_particle);
  vector<real>::type z(n_particle);
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
  vector<real>::type tmp11(n_cell);
  vector<real>::type tmp12(n_cell);
  vector<real>::type tmp13(n_cell);

  vector<size_t>::type tmp8(n_cell);
  vector<size_t>::type tmp9(n_cell);
  vector<size_t>::type tmp10(n_cell);

  /*
    initalize the positions of particles
  */
  for(int i=0; i<n_cell; ++i){
    alloc_uniform_random(
      make_cell_at(c, 0, 0, i),
      n_particle_per_cell,
      thrusting::advance(n_particle_per_cell*i, x.begin()), 
      thrusting::advance(n_particle_per_cell*i, y.begin()), 
      thrusting::advance(n_particle_per_cell*i, z.begin()), 
      i); 
  }

  /*
    add velocity of -1 toward the wall at x=0
  */
  thrust::fill(
    w.begin(),
    w.end(),
    real(-1));

  size_t step = 1000;
  real dt = real(1) / step;
  for(size_t i=0; i<500; ++i){
    std::cout << "time: " << dt * i << std::endl;

    std::cout << "make cell idx" << std::endl;
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));
  
    // std::cout << thrusting::make_list(z) << std::endl;
    // std::cout << thrusting::make_list(idx) << std::endl;

    std::cout << "sorting" << std::endl;
    thrust::sort_by_key(
      idx.begin(), idx.end(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));

    // std::cout << make_list(idx) << std::endl;
    // std::cout << make_list(x) << std::endl;

    // std::cout << make_list(x) << std::cout;
    /*
      processed by BPH routine
    */
    std::cout << "bph" << std::endl;
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
      tmp1.begin(), tmp2.begin(), tmp3.begin(), tmp4.begin(), tmp5.begin(), 
      tmp6.begin(), tmp7.begin(), tmp11.begin(), tmp12.begin(), tmp13.begin(),
      // int tmp
      tmp8.begin(), tmp9.begin(),
      i); // seed 
  
    /*
      Move
    */
    std::cout << "move" << std::endl;
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin(), m_it), // input
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin()), // output 
      make_runge_kutta_1_functor(
        make_no_force_generator(),
        dt));

    /*
      x boundary treatment
    */
    std::cout << "x boundary" << std::endl;
    thrusting::transform_if(
      n_particle,
      x.begin(),
      x.begin(), // stencil
      x.begin(), // output
      make_retrieve_less_functor(0, 1),
      thrusting::bind2nd(
        thrust::less<real>(), real(0)));

    thrusting::transform_if(
      n_particle,
      x.begin(),
      x.begin(), // stencil
      x.begin(),
      make_retrieve_greater_functor(0, 1),
      thrusting::bind2nd(
        thrust::greater<real>(), real(1)));
    
    /*
      y boundary treatment
    */
    std::cout << "y boundary" << std::endl;
    thrusting::transform_if(
      n_particle,
      y.begin(),
      y.begin(), // stencil
      y.begin(), // output
      make_retrieve_less_functor(0, 1),
      thrusting::bind2nd(
        thrust::less<real>(), real(0)));

    thrusting::transform_if(
      n_particle,
      y.begin(),
      y.begin(), // stencil
      y.begin(),
      make_retrieve_greater_functor(0, 1),
      thrusting::bind2nd(
        thrust::greater<real>(), real(1)));

    /*
      if z < 0 then w -= w
    */
    std::cout << "w -= w" << std::endl;
    thrusting::transform_if(
      n_particle,
      w.begin(), // input
      z.begin(), // stencil,
      w.begin(), // output,
      thrust::negate<real>(),
      thrusting::bind2nd(
        thrust::less<real>(),
        real(z_origin)));

    /*
      if z < 0 then z -= z
    */
    std::cout << "z -= z" << std::endl;
    thrusting::transform_if(
      n_particle,
      z.begin(), // input
      z.begin(), // stencil
      z.begin(), // output
      make_mirroring_functor(z_origin),
      thrusting::bind2nd(
        thrust::less<real>(),
        real(z_origin)));

  } // END for 
  
  /*
    density calculation
  */

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

  bucket_indexing(
    n_particle,
    idx.begin(),
    n_cell,
    tmp8.begin(),
    tmp9.begin());

  FILE *fp = fopen(filename, "w");
  for(size_t i=0; i<n_cell; ++i){
    size_t x = tmp9[i];
    fprintf(fp, "%d\n", x);
  }
  fclose(fp);
}
