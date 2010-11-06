#include "util.h"

#include <bphcuda/sort_by_cellind1.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

using namespace bphcuda;

void test_sort_by_key(){
  // TESTING of sort_by_key
  Int key[] = {2,1};
  thrust::device_vector<Int> d_key(key, key+2);
  Int value[] = {3,4};
  thrust::device_vector<Int> d_value(value, value+2);
  
  // sorting [(key, value)] by [key]
  thrust::sort_by_key(
    d_key.begin(), d_key.end(),
    thrust::make_zip_iterator(thrust::make_tuple(d_key.begin(), d_value.begin())));
  Int ans_key[] = {1,2};
  thrust::device_vector<Int> d_ans_key(ans_key, ans_key+2);
  Int ans_value[] = {4,3};
  thrust::device_vector<Int> d_ans_value(ans_value, ans_value+2);

  ASSERT_EQUAL(d_key, d_ans_key);
  ASSERT_EQUAL(d_value, d_ans_value);
}

void test_sort_by_cellind1(){
  Cell c = mk_cell(mk_real3(0,0,0), mk_real3(1,1,1), mk_int3(2,1,1));
  thrust::device_vector<Real3> d_key;
  d_key.push_back(mk_real3(1.5, 0.5, 0.5));
  d_key.push_back(mk_real3(0.5, 0.5, 0.5));
  
  thrust::device_vector<Int> d_value;
  d_value.push_back(3);
  d_value.push_back(4);
 
  sort_by_cellind1(
    d_key.begin(), d_key.end(),
    c,
    thrust::make_zip_iterator(thrust::make_tuple(d_key.begin(), d_value.begin())));
  
  thrust::device_vector<Real3> d_ans_key;
  d_ans_key.push_back(mk_real3(0.5, 0.5, 0.5));
  d_ans_key.push_back(mk_real3(1.5, 0.5, 0.5));
  
  thrust::device_vector<Int> d_ans_value;
  d_ans_value.push_back(4);
  d_ans_value.push_back(3);
  
  ASSERT_EQUAL(d_key, d_ans_key);
  ASSERT_EQUAL(d_value, d_ans_value);
}

int main(void){
  test_sort_by_key(); 
  test_sort_by_cellind1();
  return 0;
}
