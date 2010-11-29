#include <bphcuda/relax_particle_parallel.h>


#include <thrusting/real.h>
#include <thrusting/vector.h>

namespace {
  using namespace thrusting;
}

TEST(RelaxParticleParallel, Test){
  real s = 2; 
  real m = 2; // so that 1/2 * m = 1

  size_t n_particle = 3;
  
  real _u[] = {1, 1, 1}; vector<real::type> u(_u, _u+n_particle);
  real _v[] = {1, 1, 1}; vector<real::type> v(_v, _v+n_particle);
  real _w[] = {1, 1, 1}; vector<real::type> w(_w, _w+n_particle);
  real in_e[] = {17, 4, 10}; vector<real>::type in_e(_in_e, _in_e+n_particle);
  
  size_t _idx[] = {1,1,2}; vector<size_t>::type idx(_idx, _idx+n_particle);

  /*
    calc the total energy
  */
  e_kin_1_before = calc_kinetic_e();
  e_kin_2_before = calc_kinetic_e();
   
  /*
    relax 
  */
  relax_particle_parallel();

  /*
    conserving the total energy 
  */
  e_kin_1_after = calc_kinetic_e();
  e_kin_2_after = calc_kinetic_e();

  /*
    momentum is 0
  */
  momentum_1 = 
  
  /* 
    assert the in_e is unique for each cell
  */
  real _ans_in_e[] = {8, 4, 4}; vector<real>::type ans_in_e(_ans_in_e, _ans_in_e+n_particle);
  EXPECT_EQ(
    make_list(ans_in_e),
    make_list(in_e));
}

