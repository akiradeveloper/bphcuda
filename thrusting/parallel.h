#include "functional.h"

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/copy.h>

namespace thrusting {

template<typename A>
struct _parallel {
  Int n;
  A head;
  _parallel(Int n_, A head_)
  :n(n_), head(head_){}
  int size(){
    return n;
  }
  const A &head(){
    return head;
  }
  
  #include "parallel_operator.h"
  
  void operator<<(A from){
    thrust::copy(from, from+size(), head);
  }
};

template<typename A>
std::ostream &operator<<(const std::ostream &os, const _parallel<A> &a){
  std::string s;
  s += "[";
  for(int i=0; i<size()-1; i++){
    s += *(head()+i);
    s += ", ";
  }
  s += *(head()+size()-1);
  s += "]";
  os << s;
  return os;
}

template<typename A>
_parallel<A> parallel(Int n, A head){
  return _parallel<A>(n, head);
}

}
