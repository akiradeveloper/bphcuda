#include <bphcuda/real.h>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

template<typename Iter1, typename Iter2>
void alloc_ine(Iter1 cs_first, Iter1 cs_last, Iter2 ines, Int s){
  Real ratio = s / 3.0F;
  transform(
    // WRONG implementation. Function is not assigned
    thrust::make_transform_iterator(cs_first),
    thrust::make_transform_iterator(cs_last),
    thrust::make_constant_iterator(ratio),
    ines,
    thrust::multiplies<Real>());
}
