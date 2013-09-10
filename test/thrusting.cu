#include <gtest/gtest.h>

#include "thrusting/functional.h"
#include "thrusting/copy.h"
#include "thrusting/equal.h"
#include "thrusting/engine.h"
#include "thrusting/reduce_by_bucket.h"
#include "thrusting/distribution.h"
#include "thrusting/constant.h"
#include "thrusting/tuple.h"
#include "thrusting/stride_iterator.h"
#include "thrusting/time.h"
#include "thrusting/random.h"
#include "thrusting/real.h"
#include "thrusting/zip_iterator.h"
#include "thrusting/reduce_by_key.h"
#include "thrusting/iterator_equal.h"
#include "thrusting/vector.h"
#include "thrusting/assert.h"
#include "thrusting/iterator.h"
#include "thrusting/partition.h"
#include "thrusting/transform_if.h"
#include "thrusting/remove.h"
#include "thrusting/scatter.h"
#include "thrusting/bucket_indexing.h"
#include "thrusting/list.h"
#include "thrusting/generate.h"
#include "thrusting/gather.h"
#include "thrusting/transform.h"
#include "thrusting/permutation_iterator.h"
#include "thrusting/sort_by_key.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
