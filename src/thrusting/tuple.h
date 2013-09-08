#pragma once

/*
  You can 
 
  make_string(tuple)
  ostream << tuple
  tuple == tuple and tuple != tuple
  tupleN<datatype>::type and make_tupleN<datatype>(x, y) for any N and datatype

  These features are required by googletest.
*/

#include <thrust/tuple.h>

#include "tuple/detail/make_string/make_string.h"
#include "tuple/detail/n_typedef/n_typedef.h"

/*
  operator overloadings
*/
#include "op/tuple/detail/equality.h"
#include "op/tuple/detail/ostream.h"
