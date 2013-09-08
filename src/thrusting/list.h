#pragma once

/*
  List is a simple abstraction [it_begin, it_end).
  
  You can create the instance in several ways includes,
  thrusting::make_list(len, begin)
  thrusting::make_list(std::vector or host_vector or device_vector)
  
  With List instance, you can use googletest to check your module.
*/

#include "list/detail/def_list.h"
#include "list/detail/make_list.h"
#include "list/detail/make_string.h"

#include "op/list/detail/ostream.h"
