#pragma once

#include <thrusting/algorithm/partition.h>
#include <thrusting/functional.h>

#include <thrusting/vector.h>
#include <thrusting/list.h>

#include <gtest/gtest.h>

struct PARTITION_IS_ODD :public thrust::unary_function<int, bool> {
  __host__ __device__
  bool operator()(int x) const {
    return (x % 2) == 1;
  }
};

TEST(Partition, Test){
  int _xs[] = {1,0,1};
  thrusting::vector<int>::type xs(_xs, _xs+3);
  
  thrusting::partition(
    3,
    xs.begin(),
    PARTITION_IS_ODD());
}
