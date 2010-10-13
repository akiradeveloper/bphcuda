#pragma once

#include <bphcuda/cell.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

namespace bphcuda {

// sort any type of values attributing to particles in cellind1 ordering 
template<typename Iter1, typename Iter2>
void sort_by_cellind1(Iter1 xs_first, Iter1 xs_last, Cell& c, Iter2 sorted_first){
  thrust::stable_sort_by_key(
    make_transform_iterator(xs_first, bphcuda::calc_cellind1(c)),
    make_transform_iterator(xs_last, bphcuda::calc_cellind1(c)),
    sorted_first);
}

struct order_by_cellind1 :public thrust::binary_function<Real3, Real3, 

template<typename Iter>
void sort_by_cellind1(Iter xs_first, Iter xs_last, Cell& c){
  thrust::stable_sort(xs_first, xs_last)

} // end of bphcuda

