/*
Assertion for NVCC, A CUDA Compiler
*/

#include <iostream>
#include <string>

#define CHECK(BOOL, MESSAGE) bphcuda::assert((BOOL), __FILE__, __LINE__, (MESSAGE))
#define CHECK_EQUAL_ARRAY(ARY1, ARY2) bphcuda::assert_equal_array((ARY1), (ARY2), __FILE__, __LINE__)
#define CHECK_EQUAL_VALUE3(VALUE1, VALUE3) bphcuda::assert_equal_value3((VALUE1), (VALUE2), __FILE__, __LINE__)
#define CHECK_EQUAL_VALUE(VALUE1, VALUE2) bphcuda::assert_equal_value((VALUE1), (VALUE2), __FILE__, __LINE__)

namespace bphcuda {

void assert_bool(bool is_true, std::string file, int line, std::string message=""){
  if(! is_true){
    std::cout << message << std::endl;
    std::cout << "filename: " << file << std::endl;
    std::cout << "line: " << line << std::endl;
    throw "Assertion False";
  }
} 

template<typename T>
void assert_equal_value(T x, T y, std::string filename, int line){
}

template<typename T>
void assert_equal_value3(T x, T y, std::string filename, int line){
}

template<typename ArrayIterator1, typename ArrayIterator2>
void assert_equal_length(int size1, int size2, std::string filename, int line){
  assert_bool(size1==size2, filename, line, "Not the same length");
}

template<typename Array1, typename Array2>
void assert_equal_array(Array1 array1, Array2 array2, string filename, int line){
  assert_equal_array(array1.begin(), array1.end(), array2.begin(), array2.end(), filename, line); 
}

template<typename ArrayIterator1, typename ArrayIterator2>
void assert_equal_array(ArrayIterator1 first1, ArrayIterator1 last1, ArrayIterator2 first2, ArrayIterator last2, std::string filename, int line){
  int size1 = last1 - first1;
  int size2 = last2 - first2;
  assert_equal_length(size1, size2);
  for(int i=0; i<size1; i++){
    assert_bool(first1[i]==first2[i], filename, line, "Not the same array");
  }
}

} // end of bphcuda
