#pragma once

#include <thrust/sort.h>

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/vectorspace.h>
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
#include <bphcuda/alloc_in_e.h>

#include <cstdio>
#include <cstdlib>

#define SHOCKTUBE_TIME 1

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

int main(int narg, char **args){
  const size_t N = atoi(args[1]);
  const size_t M = atoi(args[2]);
  const real s = atof(args[3]);
  const real fin = atof(args[4]);
  char *plotfile = args[5];
  char *timefile = args[6];

  const size_t n_particle = (1+8) * N * M;
  const size_t n_cell = 2*M;

  const real m = 1;
  thrust::constant_iterator<real> m_it(m);

  cell c(real3(0,0,0), real3(real(1)/(n_cell), 1, 1), tuple3<size_t>::type(n_cell, 1, 1));

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
    left. high density
  */
  for(size_t i=0; i<M; ++i){
    alloc_uniform_random(
      make_cell_at(c, i, 0, 0),
      8*N,
      thrusting::advance(8*N*i, x.begin()), 
      thrusting::advance(8*N*i, y.begin()), 
      thrusting::advance(8*N*i, z.begin()), 
      i); 
  }

  /*
    right. low density
  */
  const size_t sep = 8*N*M;
  for(size_t i=M; i<2*M; ++i){
    const size_t ii = i - M;
    alloc_uniform_random(
      make_cell_at(c, i, 0, 0),
      N,
      thrusting::advance(sep + N*ii, x.begin()), 
      thrusting::advance(sep + N*ii, y.begin()), 
      thrusting::advance(sep + N*ii, z.begin()), 
      i);
  }
  
  alloc_shell_rand(
    n_particle,
    u.begin(),
    v.begin(),
    w.begin(),
    2);

  /*
    veloc in left 
  */
  const real veloc_high = sqrt(3);
  thrusting::transform(
    8*N*M,
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), veloc_high)); 
  
  const real veloc_low = sqrt(3) / sqrt(1.25);
  thrusting::transform(
    N*M,
    thrusting::advance(
      sep,
      thrusting::make_zip_iterator(
        u.begin(),
        v.begin(),
        w.begin())),
    thrusting::advance(
      sep,
      thrusting::make_zip_iterator(
        u.begin(),
        v.begin(),
        w.begin())),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), veloc_low)); 

  /*
    sorting
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
  
  /*
    allocate the in_e initially
  */
  alloc_in_e(
    n_particle,
    u.begin(), v.begin(), w.begin(),
    m,
    in_e.begin(),
    s,
    idx.begin(),
    n_cell,
    tmp1.begin(), tmp2.begin(),
    tmp8.begin(), tmp9.begin());

  stopwatch sw_idx("idx");
  stopwatch sw_sort_by_key("sort_by_key");
  stopwatch sw_bph("bph");
  stopwatch sw_move("move");
  stopwatch sw_boundary("boundary");
    
  const real dt = real(1) / n_cell;
  const size_t end_step = ( (real) fin / dt );
 
  for(size_t i=0; i<end_step; ++i){
    std::cout << "step:" << i << std::endl;
    std::cout << "time:" << dt*i << std::endl;

  #if SHOCKTUBE_TIME
    sw_idx.begin();
  #endif
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));
  #if SHOCKTUBE_TIME
    sw_idx.end();
  #endif

  #if SHOCKTUBE_TIME
    sw_sort_by_key.begin();
  #endif 
    thrust::sort_by_key(
      idx.begin(), idx.end(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));
  #if SHOCKTUBE_TIME
    sw_sort_by_key.end();
  #endif

    /*
      processed by BPH routine
    */
  #if SHOCKTUBE_TIME
    sw_bph.begin();
  #endif
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
  #if SHOCKTUBE_TIME
    sw_bph.end();
  #endif
  
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

    /*
      y boundary treatment
    */
    sw_boundary.begin();
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

    /*
      if x > 1 then u -= u
    */
    thrusting::transform_if(
      n_particle,
      u.begin(), // input
      x.begin(), // stencil,
      u.begin(), // output,
      thrust::negate<real>(),
      thrusting::bind2nd(
        thrust::greater<real>(),
        real(1)));

    /*
      if x > 1 then x -= x
    */
    thrusting::transform_if(
      n_particle,
      x.begin(), // input
      x.begin(), // stencil
      x.begin(), // output
      make_mirroring_functor(1),
      thrusting::bind2nd(
        thrust::greater<real>(),
        real(1)));
    sw_boundary.end();
  } // END for 
  
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
    /*
      normalized by the number of particles in
      high pressure cells in initial state.
    */
    real x = ((real)tmp9[i]) / (8*N);
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
