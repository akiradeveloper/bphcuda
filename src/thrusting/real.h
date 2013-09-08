#pragma once

/*
  You can switch the hidden type of 
  
  thrusting::real -> double

  only by 

  -D THRUSTING_USING_DOUBLE_FOR_REAL

  Otherwise, 

  thrusting::real -> float.

  This is very effective to build a portable software.
*/

#include "real/detail/real_typedef.h"
#include "real/detail/tuple/real.h"
