#pragma once

#include "tuple.h"

/*
  Vector space arithmatics,
  ragarding tuples are n-vector instances.
  
  Example,
  (a,b) (+|-) (c,d) -> (a+c,b+d)
  c * (a,b) -> (c*a,c*b)
  (a,b) / c -> (a/c,b/c) 

  For any length of tuple.
*/

#include "op/tuple/detail/n_operator.h"
#include "op/tuple/detail/operator.h"
