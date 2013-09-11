#pragma once

#include <thrust/advance.h>

namespace thrusting {

template<typename Distance, typename Iterator>
Iterator advance(Distance distance, Iterator it){
  thrust::advance(it, distance);
  return it;
}

} // END thrusting
