#pragma once

/*
Assertion for NVCC, A CUDA Compiler
*/

#include <assert.h>
#include <iostream>
#include <string>

#define CHECK(BOOL) bphcuda::assert_true((BOOL), __FILE__, __LINE__)
#define CHECK_EQUAL_VALUE(VALUE1, VALUE2) bphcuda::assert_equal_value((VALUE1), (VALUE2), __FILE__, __LINE__)
#define CHECK_EQUAL_VALUE3(VALUE1, VALUE2) bphcuda::assert_equal_value3((VALUE1), (VALUE2), __FILE__, __LINE__)
#define CHECK_EQUAL_ARRAY(ARY1, ARY2) bphcuda::assert_equal_array((ARY1), (ARY2), __FILE__, __LINE__)

namespace bphcuda {

void assert_bool(bool is_true, std::string filename, int line, std::string message=""){
  if(! is_true){
    std::cout << message << std::endl;
    std::cout << "filename: " << filename << std::endl;
    std::cout << "line: " << line << std::endl;
    assert(is_true);
  }
} 

void assert_true(bool is_true, std::string filename, int line){
  assert_bool(is_true, filename, line, "Not true");
}

template<typename T>
void assert_equal_value(T x, T y, std::string filename, int line){
  assert_bool(x==y, filename, line, "Not the same value");
}

template<typename T>
bool are_equal_value3(T x, T y){
  return (x.x == y.x) & (x.y == y.y) & (x.z == y.z);
}

template<typename T>
void assert_equal_value3(T x, T y, std::string filename, int line){
  assert_bool(are_equal_value3(x, y), filename, line, "Not the same value3");
}

void assert_equal_length(int size1, int size2, std::string filename, int line){
  assert_bool(size1==size2, filename, line, "Not the same length");
}

template<typename Iterator1, typename Iterator2>
void assert_equal_array(Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, std::string filename, int line){
  int size1 = last1 - first1;
  int size2 = last2 - first2;
  assert_equal_length(size1, size2, filename, line);
  for(int i=0; i<size1; i++){
    assert_bool(first1[i]==first2[i], filename, line, "Not the same array");
  }
}

template<typename Array1, typename Array2>
void assert_equal_array(Array1 array1, Array2 array2, std::string filename, int line){
  assert_equal_array(array1.begin(), array1.end(), array2.begin(), array2.end(), filename, line); 
}

} // end of bphcuda
