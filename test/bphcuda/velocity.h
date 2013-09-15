#pragma once

#include <thrusting/list.h>
#include <thrusting/vector.h>
#include <thrusting/real.h>

#include <bphcuda/velocity.h>

#include <gtest/gtest.h>

namespace {
  using namespace thrusting;
}

TEST(Velocity, MinusAverageVelocity){
  size_t n_particle = 3;
  size_t n_cell = 4;
  
  size_t _idx[] = {1,1,3}; vector<size_t>::type idx(_idx, _idx + n_particle);

  real _u[] = {1,1,2}; vector<real>::type u(_u, _u + n_particle);
  real _v[] = {3,3,4}; vector<real>::type v(_v, _v + n_particle);
  real _w[] = {5,5,6}; vector<real>::type w(_w, _w + n_particle);

  vector<real>::type ave_u(n_cell);
  vector<real>::type ave_v(n_cell);
  vector<real>::type ave_w(n_cell);
 
  real _ans_ave_u[] = {0,1,0,2}; vector<real>::type ans_ave_u(_ans_ave_u, _ans_ave_u + n_cell);
  real _ans_ave_v[] = {0,3,0,4}; vector<real>::type ans_ave_v(_ans_ave_v, _ans_ave_v + n_cell);
  real _ans_ave_w[] = {0,5,0,6}; vector<real>::type ans_ave_w(_ans_ave_w, _ans_ave_w + n_cell);
  
  vector<size_t>::type tmp1(n_cell);
  vector<size_t>::type tmp2(n_cell);
  vector<real>::type tmp3(n_cell);
  vector<real>::type tmp4(n_cell);
  vector<real>::type tmp5(n_cell);

  bphcuda::minus_average_velocity(
    n_particle,
    u.begin(), v.begin(), w.begin(),
    idx.begin(),
    n_cell,
    ave_u.begin(), ave_v.begin(), ave_w.begin(),
    tmp3.begin(), tmp4.begin(), tmp5.begin(),
    tmp1.begin(), tmp2.begin());

  EXPECT_EQ(
    make_list(ans_ave_u),
    make_list(ave_u));

  EXPECT_EQ(
    make_list(ans_ave_v),
    make_list(ave_v));
     
  EXPECT_EQ(
    make_list(ans_ave_w),
    make_list(ave_w));
}

TEST(Velocity, PlusAverageVelocity){
  size_t n_particle = 3;
  size_t n_cell = 4;
  
  size_t _idx[] = {1,1,3}; vector<size_t>::type idx(_idx, _idx + n_particle);

  vector<real>::type u(n_particle);
  vector<real>::type v(n_particle);
  vector<real>::type w(n_particle);
 
  real _ave_u[] = {0,1,0,2}; vector<real>::type ave_u(_ave_u, _ave_u + n_cell);
  real _ave_v[] = {0,3,0,4}; vector<real>::type ave_v(_ave_v, _ave_v + n_cell);
  real _ave_w[] = {0,5,0,6}; vector<real>::type ave_w(_ave_w, _ave_w + n_cell);

  bphcuda::plus_average_velocity(
    n_particle,
    u.begin(), v.begin(), w.begin(),
    idx.begin(),
    n_cell,
    ave_u.begin(), ave_v.begin(), ave_w.begin());

  real _ans_u[] = {1,1,2}; vector<real>::type ans_u(_ans_u, _ans_u + n_particle);
  real _ans_v[] = {3,3,4}; vector<real>::type ans_v(_ans_v, _ans_v + n_particle);
  real _ans_w[] = {5,5,6}; vector<real>::type ans_w(_ans_w, _ans_w + n_particle);

  EXPECT_EQ(
    make_list(ans_u),
    make_list(u));

  EXPECT_EQ(
    make_list(ans_v),
    make_list(v));
     
  EXPECT_EQ(
    make_list(ans_w),
    make_list(w));
}

