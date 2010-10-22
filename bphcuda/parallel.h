#include <bphcuda/int.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>

namespace bphcuda {

template<typename A>
struct _parallel {
  Int n;
  A head;
  _parallel(Int n_, A head_)
  :n(n_), head(head_){}
  void operator+(A with){
    typedef typename thrust::iterator_value<A>::type VALUE_TYPE;
    thrust::transform(head, head+n, with, head, thrust::plus<VALUE_TYPE>());
  }
};

template<typename A>
_parallel<A> parallel(Int n, A head){
  return _parallel<A>(n, head);
}

}
