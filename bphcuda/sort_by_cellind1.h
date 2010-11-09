#pragma once

#include <bphcuda/cell.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

/*
  deprecated
*/
namespace bphcuda {

/*
Output : [TupleN] of same length

The length of ps and target list are the same.
Typically zipped list of to-be-sorted lists.
*/
template<typename Position, typename Target>
void sort_by_cellind1(Position ps_F, Position ps_L, const Cell &c, Target list_F){
  thrust::sort_by_key(
    thrust::make_transform_iterator(ps_F, calc_cellind1(c)),
    thrust::make_transform_iterator(ps_L, calc_cellind1(c)),
    list_F);
}
 
} // end of bphcuda

