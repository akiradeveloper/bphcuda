#pragma once

#include <bphcuda/cell.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

namespace bphcuda {

/*
the length of ps and target list are the same.
typically zipped list of to-be-sorted lists.
*/
template<typename Iter1, typename Iter2>
void sort_by_cellind1(Iter1 ps_first, Iter1 ps_last, Cell &c, Iter2 target_first){
  thrust::sort_by_key(
    thrust::make_transform_iterator(ps_first, calc_cellind1(c)),
    thrust::make_transform_iterator(ps_last, calc_cellind1(c)),
    target_first);
}
 
} // end of bphcuda

