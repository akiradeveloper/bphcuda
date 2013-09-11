#pragma once

#include <iostream>
#include <string>

#ifndef THRUSTING_PRETTY_PRINT_DISABLED
#define THRUSTING_PP(MSG, OBJ) thrusting::pp((MSG), (OBJ), __FILE__, __LINE__)
#else
#define THRUSTING_PP(MSG, OBJ) do {} while(0)
#endif

namespace thrusting
{

template<
typename Object>
void pp(std::string msg, Object obj, std::string filename, size_t lineno)
{
    std::cout << filename << " " << "L" << lineno << std::endl;
    std::cout << msg << " : " << obj << std::endl;
}

} /* END thrusting */
