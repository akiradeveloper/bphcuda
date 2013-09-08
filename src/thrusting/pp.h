#pragma once

#include <iostream>
#include <string>

#define THRUSTING_PP(MSG, OBJ) thrusting::pp((MSG), (OBJ), __FILE__, __LINE__)

namespace thrusting {

template<
typename Object>
void pp(std::string msg, Object obj, std::string filename, size_t lineno){
#ifndef THRUSTING_PRETTY_PRINT_DISABLED
  std::cout << filename << " " << "L" << lineno << std::endl;
  std::cout << msg << " : " << obj << std::endl;
#endif
}

} // END thrusting
