#pragma once

#include "util.h"

#include <iostream>
#include <string>

#define ASSERT_TRUE(BOOL) bphcuda::assert((BOOL), __FILE__, __LINE__)
#define ASSERT_EQUAL(X, Y) bphcuda::assert_equal((X), (Y), __FILE__, __LINE__)

namespace bphcuda {

void assert(bool is_true, const std::string &filename, int lineno){
  unless(is_true){
    std::cout << "-----ASSERT FALSE-----" << std::endl;
    std::cout << "filename: " << filename << std::endl;
    std::cout << "lineno: " << lineno << std::endl;
  }
} 

template<typename T1, typename T2>
void assert_equal(T1 x, T2 y, std::string filename, int lineno){
  unless(x == y){
    std::cout << "-----ASSERT FALSE-----" << std::endl;
    std::cout << "left  hand: " << x << std::endl;
    std::cout << "right hand: " << y << std::endl;
    std::cout << "filename: " << filename << std::endl;
    std::cout << "lineno: " << lineno << std::endl;
  }
}

} // end of bphcuda
