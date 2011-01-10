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

namespace {
  using namespace thrusting;
  using namespace bphcuda;
}

namespace bphcuda {

  __host__ __device__
  real get_distance(const real3 &p, const real3 & q){
    real x = get<0>(p) - get<0>(q);
    real y = get<1>(p) - get<1>(q);
    real z = get<2>(p) - get<2>(q);
    return thrusting::sqrtr(x*x + y*y + z*z);
  }

  __host__ __device__
  real3 get_normal_velocity(const real3 &p, const real3 &toward){
    real3 vec = p - toward;
    real len = get_distance(p, toward);
    return vec / len;
  }

  __host__ __device__
  real3 get_center_of(cell &c){
    real x = real(0.5) * (c.x_min() + c.x_max());
    real y = real(0.5) * (c.y_min() + c.y_max());
    real z = real(0.5) * (c.z_min() + c.z_max());

    return real3(x, y, z);
  }
  
  class alloc_normal_velocity_functor :public thrust::unary_function<real3, real3> {
    real3 _center;
  public:
    alloc_normal_velocity_functor(real3 center)
    :_center(center){};
    __host__ __device__
    real3 operator()(const real3 &p) const {
      return get_normal_velocity(p, _center);
    }
  };

}

int main(int narg, char **args){
  const char *filename = args[1];
  const size_t n_particle_by_cell = atoi(args[2]);
 
  /*
    half of the cell size
  */
  const size_t size = 50;

  const size_t n_cell = (2*size) * (2*size);
  const size_t n_particle = n_cell * n_particle_by_cell; 

  cell c = make_cell(
    real3(0,0,0),
    real3(1/size, 1/size, 1),
    tuple3<size_t>::type(2*size,2*size,1));

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
    alloc random positions
  */
  for(size_t i=0; i<size*2; ++i){
    for(size_t j=0; j<size*2; ++j){
      size_t ind = i*size*2 + j;
      alloc_uniform_random(
        make_cell_at(c, i, j, 0),
        n_particle_by_cell,
        thrusting::advance(ind*n_particle_by_cell, x.begin()),
        thrusting::advance(ind*n_particle_by_cell, y.begin()),
        thrusting::advance(ind*n_particle_by_cell, z.begin()),
        i);
    }
  }

  /*
    alloc velocity toward the center
  */
  real3 center = get_center_of(c);
  thrusting::transform(
    n_particle,
    thrusting::make_zip_iterator(
      x.begin(),
      y.begin(),
      z.begin()),
    thrusting::make_zip_iterator(
      u.begin(),
      v.begin(),
      w.begin()),
    alloc_normal_velocity_functor(center));
}
