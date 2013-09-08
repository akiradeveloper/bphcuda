#pragma once

#include <iostream>

namespace thrusting {
namespace detail {

template<typename Iterator>
std::ostream &operator<<(std::ostream &os, const list<Iterator> &xs){
  return os << thrusting::detail::make_string(xs);
}

} // END detail
} // END thrusting
