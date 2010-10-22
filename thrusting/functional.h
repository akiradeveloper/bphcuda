#include <thrust/functional.h>

template<typename A, typename B>
struct left_shift :public thrust::binary_function<A, B, A> {
  A operator()(A x, B y){
    return x << y;
  }
};

template<typename A, typename B>
struct right_shift :public thrust::binary_function<A, B, A> {
  A operator()(A x, B y){
    return x >> y;
  }
};
