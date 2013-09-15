#include <gtest/gtest.h>

#include "bphcuda/streaming.h"
#include "bphcuda/cell.h"
#include "bphcuda/uniform_random.h"
#include "bphcuda/momentum.h"
#include "bphcuda/velocity.h"
#include "bphcuda/maxwell_distribution.h"
#include "bphcuda/shell_distribution.h"
#include "bphcuda/total_e.h"
#include "bphcuda/relax_cell.h"
#include "bphcuda/kinetic_e.h"
#include "bphcuda/relax.h"
#include "bphcuda/alloc_in_e.h"
#include "bphcuda/force.h"
#include "bphcuda/real_comparator.h"
#include "bphcuda/boundary.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
