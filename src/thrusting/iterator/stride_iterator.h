#pragma once

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace thrusting {

namespace detail {
template<typename Idx>
class stride_functor :public thrust::unary_function<Idx, Idx> {
  Idx _first, _step;
public:
  stride_functor(Idx first, Idx step)
  :_first(first), _step(step){}
  __host__ __device__
  Idx operator()(Idx idx) const {
    return _first + idx * _step;
  }
};
} // END detail

template<typename Idx>
detail::stride_functor<Idx> make_stride_functor(Idx first, Idx step){
  return detail::stride_functor<Idx>(first, step);
}

/*
  Make fancy iterator that generates
  [first, first+step, first+2*step, ...] lazily.

  Use this iterator
  to alloc randomly generated tuples to an array.
*/
template<typename Idx>
thrust::transform_iterator<detail::stride_functor<Idx>, thrust::counting_iterator<Idx> >
make_stride_iterator(Idx first, Idx step){
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    make_stride_functor(first, step));
}

} // END thrusting
