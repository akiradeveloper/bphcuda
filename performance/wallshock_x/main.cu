#pragma once

#include <thrust/sort.h>

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/real.h>
#include <thrusting/functional.h>
#include <thrusting/time.h>

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

  const size_t n_particle_per_cell = atoi(args[1]);
  const size_t n_cell = atoi(args[2]);
  const real s = atof(args[3]);
  const real time = atof(args[4]);
  char *plotfile = args[5];
  char *timefile = args[6];
  
  const size_t n_particle = n_particle_per_cell * n_cell;

  real m = 1;
  thrust::constant_iterator<real> m_it(m);

  cell c(real3(0,0,0), real3(real(1)/n_cell, 1, 1), tuple3<size_t>::type(n_cell, 1, 1));

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
  for(size_t i=0; i<n_cell; ++i){
    alloc_uniform_random(
      make_cell_at(c, i, 0, 0),
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
    u.begin(),
    u.end(),
    real(-1));

  stopwatch sw_idx("idx");
  stopwatch sw_sort_by_key("sort_by_key");
  stopwatch sw_bph("bph");
  stopwatch sw_move("move");
  stopwatch sw_boundary("boundary");

  // dt = dx
  real dt = real(1) / n_cell;
  const size_t max_step = time / dt;

  for(size_t i=0; i<max_step; ++i){
    std::cout << "step:" << i << std::endl;
    std::cout << "time:" << dt*i << std::endl;

    sw_idx.begin();
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));
    sw_idx.end();

    sw_sort_by_key.begin();
    thrust::sort_by_key(
      idx.begin(), idx.end(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));
    sw_sort_by_key.end();

    /*
      processed by BPH routine
    */
    sw_bph.begin();
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
    sw_bph.end();
  
    /*
      Move
    */
    sw_move.begin();
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin(), m_it), // input
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin(), u.begin(), v.begin(), w.begin()), // output 
      make_runge_kutta_1_functor(
        make_no_force_generator(),
        dt));
    sw_move.end();

    sw_boundary.begin();
    /*
      y boundary treatment
    */
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
      z boundary treatment
    */
    thrusting::transform_if(
      n_particle,
      z.begin(),
      z.begin(), // stencil
      z.begin(), // output
      make_retrieve_less_functor(0, 1),
      thrusting::bind2nd(
        thrust::less<real>(), real(0)));

    thrusting::transform_if(
      n_particle,
      z.begin(),
      z.begin(), // stencil
      z.begin(),
      make_retrieve_greater_functor(0, 1),
      thrusting::bind2nd(
        thrust::greater<real>(), real(1)));

    /*
      if x < 0 then u -= u
    */
    thrusting::transform_if(
      n_particle,
      u.begin(), // input
      x.begin(), // stencil,
      u.begin(), // output,
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
      x.begin(), // stencil
      x.begin(), // output
      make_mirroring_functor(0),
      thrusting::bind2nd(
        thrust::less<real>(),
        real(0)));
     sw_boundary.end();
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
  
  // density data
  FILE *fp = fopen(plotfile, "w");
  for(size_t i=0; i<n_cell; ++i){
    real x = ((real)tmp9[i]) / n_particle_per_cell;
    fprintf(fp, "%f\n", x);
  }
  fclose(fp);

  // time data
  FILE *fp2 = fopen(timefile, "w");
  fprintf(fp2, "idx:%f\n", sw_idx.average());
  fprintf(fp2, "sort:%f\n", sw_sort_by_key.average());
  fprintf(fp2, "bph:%f\n", sw_bph.average());
  fprintf(fp2, "move:%f\n", sw_move.average());
  fprintf(fp2, "boundary:%f\n", sw_boundary.average());
  fclose(fp2);

  return 0;
}
