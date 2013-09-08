#pragma once

#include <cassert>

#ifndef THRUSTING_ASSERTION_DISABLED
  #ifndef THRUSTING_DEBUG
    #define THRUSTING_DEBUG 
  #endif
#endif

/*
  TODO
  named check although the filename is assert right?
*/
#define THRUSTING_CHECK(BOOL) thrusting::check((BOOL))

namespace thrusting {

/*
  TODO
  temporarily using assert but,
  this should be able to be used in device code
*/
void check(bool b){
#ifdef THRUSTING_DEBUG
  assert(b);
#endif
}

} // END thrusting
  
