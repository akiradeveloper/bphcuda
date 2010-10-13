#pragma once

#include <bphcuda/macro.h>

#include <iostream>
#include <string>

#define ASSERT_TRUE(BOOL) bphcuda::assert((BOOL), __FILE__, __LINE__)

namespace bphcuda {

void assert(bool is_true, std::string filename, int lineno){
  unless(is_true){
    std::cout << "-----ASSERT FALSE-----" << std::endl;
    std::cout << "filename: " << filename << std::endl;
    std::cout << "lineno: " << lineno << std::endl;
  }
} 

} // end of bphcuda
