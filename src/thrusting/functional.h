#pragma once

/*
  functional.h in Thrusting library supports
  several useful functions such as 
  
  bind1st
  bind2nd
  curry, uncurry
  compose

  Enjoy Functional Programming!
*/

#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/iterator_traits.h>

#include <thrusting/iterator.h>

namespace thrusting {

namespace detail {
template<typename F>
class flipper :public thrust::binary_function<
typename F::second_argument_type,
typename F::first_argument_type,
typename F::result_type> {
  F _f;
public:
  flipper(F f)
  :_f(f){}
  __host__ __device__
  typename F::result_type operator()(
  const typename F::second_argument_type &b, const typename F::first_argument_type &a) const {
    return _f(a, b);
  }
};
} // END detail

/*
  flip the arguments of given binary function
  a->b->c -> b->a->c
*/
template<typename F>
detail::flipper<F> flip(F f){
  return detail::flipper<F>(f);
}

namespace detail {
template<typename F>
class binder1st :public thrust::unary_function<
  typename F::second_argument_type,
  typename F::result_type> {
  F _f;
  typename F::first_argument_type _a;
public:
  binder1st(F f, const typename F::first_argument_type &a)
  :_f(f), _a(a){}
  __host__ __device__
  typename F::result_type operator()(const typename F::second_argument_type &b) const {
    return _f(_a, b);
  }
};
} // END detail

/*
  bind1st
  a->b->c -> a -> b->c
*/
template<typename F>
detail::binder1st<F> bind1st(F f, const typename F::first_argument_type &a) {
  return detail::binder1st<F>(f, a);
}

/*
  bind2nd
  a->b->c -> b -> a->c
*/
template<typename F>
detail::binder1st<detail::flipper<F> > bind2nd(F f, const typename F::second_argument_type &b) {
  return thrusting::bind1st(flip(f), b);
}

namespace detail {
template<typename F>
class currier :public thrust::binary_function<
typename thrust::tuple_element<0, typename F::argument_type>::type, 
typename thrust::tuple_element<1, typename F::argument_type>::type, 
typename F::result_type> {
  F _f;
public:
  currier(F f)
  :_f(f){}
  __host__ __device__
  typename F::result_type operator()(
  const typename thrust::tuple_element<0, typename F::argument_type>::type &a, 
  const typename thrust::tuple_element<1, typename F::argument_type>::type &b) const {
    /*
      should be once store the tuple because it has not reference
    */
    return _f(thrust::make_tuple(a, b));
  }
};
} // END detail

/*
  (a,b)->c -> a->b->c
*/
template<typename F>
detail::currier<F> curry(F f){
  return detail::currier<F>(f);
}

namespace detail {
template<typename F>
class uncurrier :public thrust::unary_function<
thrust::tuple<typename F::first_argument_type, typename F::second_argument_type>, 
typename F::result_type> {
  F _f;
public:
  uncurrier(F f)
  :_f(f){}
  __host__ __device__
  typename F::result_type operator()(
  const thrust::tuple<typename F::first_argument_type, typename F::second_argument_type> &t) const {
    return _f(thrust::get<0>(t), thrust::get<1>(t));
  }
};  
} // END detail

/*
  a->b->c -> (a,b)->c
*/
template<typename F>
detail::uncurrier<F> uncurry(F f){
  return detail::uncurrier<F>(f);
}

namespace detail {
template<typename F, typename G>
class composer :public thrust::unary_function<typename G::argument_type, typename F::result_type> {
  F _f;
  G _g;
public:
  composer(F f, G g)
  :_f(f), _g(g){}
  __host__ __device__
  typename F::result_type operator()(const typename G::argument_type &x) const {
    typename G::result_type y = _g(x);
    typename F::result_type z = _f(y);
    return z;
  }
};
} // END detail

/*
  f * g
  b->c -> a->b -> a->c
*/
template<typename F, typename G>
detail::composer<F, G> compose(F f, G g){
  return detail::composer<F, G>(f, g);
}

namespace detail {
template<typename In, typename Out>
class constant_functor :public thrust::unary_function<In, Out> {
  Out _value;
public:
  constant_functor(Out value)
  :_value(value){} 
  __host__ __device__
  Out operator()(const In &in) const {
    return _value;
  }
};
} // END detail

template<typename In, typename Out>
detail::constant_functor<In, Out> make_constant_functor(Out value){
  return detail::constant_functor<In, Out>(value);
}

namespace detail {
/*
  a -> b -> (a*b)::b
*/
template<typename A, typename B>
struct multiplies :public thrust::binary_function<A, B, B> {
  __host__ __device__
  B operator()(const A &x, const B &y) const {
    return x * y;
  }
};
} // END detail

template<typename A, typename B>
detail::multiplies<A, B> multiplies(){
  return detail::multiplies<A, B>();
}

namespace detail {
/*
  a -> b -> (a/b)::a
*/
template<typename A, typename B>
struct divides :public thrust::binary_function<A, B, A> {
  __host__ __device__
  A operator()(const A &x, const B &y) const {
    return x / y;
  }
};
} // END detail

template<typename A, typename B>
detail::divides<A, B> divides(){
  return detail::divides<A, B>();
}

namespace detail {
/*
  a -> b -> (a<<b)::a
*/
template<typename A, typename B>
struct left_shift :public thrust::binary_function<A, B, A> {
  __host__ __device__
  A operator()(const A &x, const B &y) const {
    return x << y;
  }
};
} // END detail

template<typename A, typename B>
detail::left_shift<A, B> left_shift(){
  return detail::left_shift<A, B>();
}

namespace detail {
/*
  a -> b -> (a>>b)::a
*/
template<typename A, typename B>
struct right_shift :public thrust::binary_function<A, B, A> {
  __host__ __device__
  A operator()(const A &x, const B &y) const {
    return x >> y;
  }
};
} // END detail

template<typename A, typename B>
detail::right_shift<A, B> right_shift(){
  return detail::right_shift<A, B>();
}

} // END thrusting
