#pragma once

#include <thrust/functional.h>

namespace thrusting {

template<typename A, typename B, typename C>
struct compose :public thrust::unary_function<A, C> {
  thrust::unary_function<A, B> first;
  thrust::unary_function<B, C> second;
  compose(
    thrust::unary_function<A, B> first_,
    thrust::unary_function<B, C> second_)
  :first(first_), second(second_){}
  
  C operator()(A x, B y){
    return second(first(x));
  }
};

template<typename A, typename B, typename C>
struct multiplies :public thrust::binary_function<A, B, C> {
  C operator()(A x, B y){
    return x * y;
  }
};

template<typename A, typename B>
struct multiplies :public multiplies<A, B, A>

template<typename A, typename B, typename C>
struct divides :public thrust::binary_function<A, B, C>{
  C operator()(A x, B y){
    return x / y;
  }
};

template<typename A, typename B>
struct divides :public divides<A, B, A>

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

} // end thrusting
