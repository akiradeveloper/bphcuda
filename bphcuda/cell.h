#pragma once

#include <thrusting/real.h>
#include <thrusting/tuple.h>
#include <thrusting/functional.h>

namespace {
  using namespace thrusting;
  typedef thrust::tuple<size_t, size_t, size_t> size3;
}

namespace bphcuda {

class cell {
  real3 _origin;
  real3 _spaces;
  size3 _dims;
public:
  __host__ __device__
  cell(real3 origin, real3 spaces, size3 dims)
  :_origin(origin), _spaces(spaces), _dims(dims){}

  __host__ __device__
  real3 origin() const {
    return _origin;
  }

  __host__ __device__
  real3 spaces() const {
    return _spaces;
  }

  __host__ __device__
  real3 dims() const {
    return _dims;
  }

  __host__ __device__
  real x_min() const {
    return origin().get<0>();
  }  

  __host__ __device__
  real x_max() const {
    return origin().get<0>() + dims().get<0>() * spaces().get<0>();
  }

  __host__ __device__
  real y_min() const {
    return origin().get<1>();
  }  

  __host__ __device__
  real y_max() const {
    return origin().get<1>() + dims().get<1>() * spaces().get<1>();
  }

  __host__ __device__
  real z_min() const {
    return origin().get<2>();
  }  

  __host__ __device__
  real z_max() const {
    return origin().get<2>() + dims().get<2>() * spaces().get<2>();
  }

  __host__ __device__
  real3 origin(size_t i, size_t j, size_t k) const {
    real x = spaces().get<0>() * i;
    real y = spaces().get<1>() * j;
    real z = spaces().get<2>() * k;
    return real3(x, y, z);
  }
};

__host__ __device__
cell make_cell(real3 origin, real3 spaces, size3 dims){
  return cell(origin, spaces, dims);
}

__host__ __device__
cell make_cell_at(const cell &c, size_t i, size_t j, size_t k) {
  return make_cell(
    c.origin(i, j, k),
    c.spaces(),
    size3(1,1,1));
}

__host__ __device__
size3 calc_idx3(const cell &c, const real3 &p){
  size_t xidx = (p.get<0>()-c.origin().get<0>()) / c.spaces().get<0>();
  size_t yidx = (p.get<1>()-c.origin().get<1>()) / c.spaces().get<1>();
  size_t zidx = (p.get<2>()-c.origin().get<2>()) / c.spaces().get<2>();
  return size3(xidx, yidx, zidx);
}

__host__ __device__
size_t conv_idx3_idx1(const cell &c, const size3 &idx3){
  return idx3.get<0>() * c.dims().get<1>() * c.dims().get<2>() +
         idx3.get<1>() * c.dims().get<2>() +
         idx3.get<2>();
}

__host__ __device__
size_t calc_idx1(const cell &c, const real3 &p){
  return conv_idx3_idx1(c, calc_idx3(c, p));
}

namespace detail {
/*
  p -> idx
*/
class calc_cellidx1 :public thrust::unary_function<real3, size_t> {
  cell c;
public:
  calc_cellidx1(cell c_)
  :c(c_){}
  __host__ __device__
  size_t operator()(const real3 &p) const {
    return calc_idx1(c, p);
  }
};
} // END detail

detail::calc_cellidx1 make_cellidx1_calculator(const cell &c){
  return detail::calc_cellidx1(c);
}

} // END bphcuda
