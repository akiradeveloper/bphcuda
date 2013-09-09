#include <gtest/gtest.h>

#include "streaming.h"
#include "cell.h"
#include "uniform_random.h"
#include "momentum.h"
#include "velocity.h"
#include "maxwell_distribution.h"
#include "shell_distribution.h"
#include "total_e.h"
#include "relax_cell.h"
#include "kinetic_e.h"
#include "relax.h"
#include "alloc_in_e.h"
#include "force.h"
#include "real_comparator.h"
#include "boundary.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
