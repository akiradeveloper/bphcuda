#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <thrusting/iterator.h>
#include <thrusting/algorithm/copy.h>

#include <thrust/equal.h>
#include <thrust/advance.h>

/*
  Explanation,  

  List in Thrusting is an abstraction
  that is like vector type but
  the actual vector hidden is inquestionable.
  The equality is only asserted by the values on the vector. 

  Akira Hayakawa 2010
*/

namespace thrusting {
namespace detail {

template<typename T>
class list {
  
  typedef typename thrust::host_vector<T>::const_iterator Iterator;

  thrust::host_vector<T> _xs;

  Iterator begin() const {
    return _xs.begin();
  }

  Iterator end() const {
    Iterator it = begin();
    thrust::advance(it, length());
    return it;
  }

public:

  size_t length() const {
    return _xs.size();
  }

  /*
    DO NOT USE.
  */
  list(thrust::host_vector<T> xs)
  :_xs(xs){};

  bool operator==(const list<T> &ys) const {
    if(length() != ys.length()){ return false; }
    return std::equal(begin(), end(), ys.begin());
  }

  bool operator!=(const list<T> &ys) const {
    return !( *(this) == ys );
  }

  ::std::string to_s() const {
    std::stringstream ss;
    ss << "[";
    Iterator end = this->end(); thrust::advance(end, -1);
    for(Iterator it = begin(); it != end; thrust::advance(it, 1)){
      ss << *it;
      ss << ", ";
    }
    ss << *end;
    ss << "]";
    return ss.str();
  }
};

} // END detail

/*
  TODO
  resolve DRY
*/
template<typename Iterator>
detail::list<typename thrust::iterator_value<Iterator>::type> make_list(size_t n, Iterator head){
  thrust::host_vector<typename thrust::iterator_value<Iterator>::type> xs(n);
  thrusting::copy(n, head, xs.begin());
  return detail::list<typename thrust::iterator_value<Iterator>::type>(xs);
}

} // END thrusting
