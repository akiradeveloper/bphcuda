#pragma once

#include <thrusting/list.h>

namespace thrusting {
namespace detail {

/*
  make string format of list -> [a, b, c]
*/
template<typename Iterator>
::std::string make_string(const thrusting::detail::list<Iterator> &xs){
  // unused
//  size_t n = xs.length();
  return xs.to_s();
}
  
} // END detail
} // END thrusting
