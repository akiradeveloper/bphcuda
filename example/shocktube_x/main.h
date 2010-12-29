#pragma once

#include <thrust/sort.h>

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/vectorspace.h>
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
#include <bphcuda/alloc_in_e.h>

#include <cstdio>
#include <cstdlib>

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

int main(int narg, char **args){
  char *filename = args[1];
  const real s = atof(args[2]);
  const size_t end_step = atoi(args[3]);
  
  /*
    num for low 
  */
  const size_t n_particle_per_cell = 100;
  
  /*
    half of the cells
  */
  const size_t n_cell = 500;

  const size_t n_particle = (1+8) * n_particle_per_cell * n_cell;

  const real m = 1;
  thrust::constant_iterator<real> m_it(m);

  cell c(real3(0,0,0), real3(real(1)/(2*n_cell), 1, 1), tuple3<size_t>::type(2*n_cell, 1, 1));

  vector<real>::type x(n_particle);
  vector<real>::type y(n_particle);
  vector<real>::type z(n_particle);
  vector<real>::type u(n_particle);
  vector<real>::type v(n_particle);
  vector<real>::type w(n_particle);
  vector<real>::type in_e(n_particle);

  vector<size_t>::type idx(n_particle);

  vector<real>::type tmp1(2*n_cell);
  vector<real>::type tmp2(2*n_cell);
  vector<real>::type tmp3(2*n_cell);
  vector<real>::type tmp4(2*n_cell);
  vector<real>::type tmp5(2*n_cell);
  vector<real>::type tmp6(2*n_cell);
  vector<real>::type tmp7(2*n_cell);
  vector<real>::type tmp11(2*n_cell);
  vector<real>::type tmp12(2*n_cell);
  vector<real>::type tmp13(2*n_cell);

  vector<size_t>::type tmp8(2*n_cell);
  vector<size_t>::type tmp9(2*n_cell);
  vector<size_t>::type tmp10(2*n_cell);

  /*
    initalize the positions of particles
  */

  /*
    left. high density
  */
  for(size_t i=0; i<n_cell; ++i){
    alloc_uniform_random(
      make_cell_at(c, i, 0, 0),
      8*n_particle_per_cell,
      thrusting::advance(8*n_particle_per_cell*i, x.begin()), 
      thrusting::advance(8*n_particle_per_cell*i, y.begin()), 
      thrusting::advance(8*n_particle_per_cell*i, z.begin()), 
      0); 
  }

  /*
    right. low density
  */
  const size_t sep = 8*n_particle_per_cell*n_cell;
  for(size_t i=n_cell; i<2*n_cell; ++i){
    const size_t ii = i - n_cell;
    alloc_uniform_random(
      make_cell_at(c, i, 0, 0),
      n_particle_per_cell,
      thrusting::advance(sep + n_particle_per_cell*ii, x.begin()), 
      thrusting::advance(sep + n_particle_per_cell*ii, y.begin()), 
      thrusting::advance(sep + n_particle_per_cell*ii, z.begin()), 
      1);
  }
  // std::cout << make_list(x) << std::endl; 
  // std::cout << make_list(y) << std::endl; 
  // std::cout << make_list(z) << std::endl; 
  
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
    8*n_particle_per_cell*n_cell,
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    thrusting::bind1st(thrusting::multiplies<real, real3>(), veloc_high)); 
  
  // std::cout << make_list(u) << std::endl;

  const real veloc_low = sqrt(3) / sqrt(1.25);
  thrusting::transform(
    n_particle_per_cell*n_cell,
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
    n_cell*2,
    tmp1.begin(), tmp2.begin(),
    tmp8.begin(), tmp9.begin());
    

  // std::cout << make_list(in_e) << std::endl;

  const size_t step = 1000;
  const real dt = real(1) / step;
 
  for(size_t i=0; i<end_step; ++i){
    std::cout << "time: " << dt * i << std::endl;

    std::cout << "make cell idx" << std::endl;
    thrusting::transform(
      n_particle,
      thrusting::make_zip_iterator(x.begin(), y.begin(), z.begin()),
      idx.begin(),
      make_cellidx1_calculator(c));

    std::cout << "sorting" << std::endl;
    thrust::sort_by_key(
      idx.begin(), idx.end(),
      thrusting::make_zip_iterator(
        x.begin(), y.begin(), z.begin(),
        u.begin(), v.begin(), w.begin(),
        in_e.begin()));

    if(i==121){
//  FILE *fp = fopen(filename, "w");
//  for(size_t i=0; i<2*n_cell; ++i){
//    size_t x = tmp9[i];
//    fprintf(fp, "%d\n", x);
//  }
//  fclose(fp);
    //   std::cout << make_list(idx) << std::endl;
    //  std::cout << make_list(y) << std::endl;
    }

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
      2*n_cell,
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
      y boundary treatment
    */
    std::cout << "x boundary" << std::endl;
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
    std::cout << "z boundary" << std::endl;
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

    if(i==120){
//  FILE *fp1 = fopen("u.dat", "w");
//  for(size_t i=0; i<n_particle; ++i){
//    real x = u[i];
//    fprintf(fp1, "%f\n", x);
//  }
//  fclose(fp1);
//
//  FILE *fp2 = fopen("v.dat", "w");
//  for(size_t i=0; i<n_particle; ++i){
//    real x = v[i];
//    fprintf(fp2, "%f\n", x);
//  }
//  fclose(fp2);
    }

    /*
      if x < 0 then u -= u
    */
    std::cout << "u -= u" << std::endl;
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
    std::cout << "x -= x" << std::endl;
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
    std::cout << "u -= u" << std::endl;
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
    std::cout << "x -= x" << std::endl;
    thrusting::transform_if(
      n_particle,
      x.begin(), // input
      x.begin(), // stencil
      x.begin(), // output
      make_mirroring_functor(1),
      thrusting::bind2nd(
        thrust::greater<real>(),
        real(1)));

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
    2*n_cell,
    tmp8.begin(),
    tmp9.begin());

  FILE *fp = fopen(filename, "w");
  for(size_t i=0; i<2*n_cell; ++i){
    size_t x = tmp9[i];
    fprintf(fp, "%d\n", x);
  }

  fclose(fp);
}
