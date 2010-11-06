#include <bphcuda/cell_indexing.h>

#include "util.h"

using namespace bphcuda;

int main(void){
  int input[] = {0,0,0,1,1,3,4};
  thrust::device_vector<int> d_input(input, input+7);
  thrust::device_vector<int> prefixes(5);
  thrust::device_vector<int> sizes(5);
  
  cell_indexing(d_input.begin(), d_input.end(), prefixes.begin(), prefixes.end(), sizes.begin());     
  int ans_prefixes[] = {0,3,5,5,6};
  int ans_sizes[] = {3,2,0,1,1};
  ASSERT_EQUAL(prefixes, thrust::device_vector<int>(ans_prefixes, ans_prefixes+5));
  ASSERT_EQUAL(sizes, thrust::device_vector<int>(ans_sizes, ans_sizes+5));

  return 0;
}

