#include <gtest/gtest.h>

#include "functional.h"
#include "copy.h"
#include "equal.h"
#include "engine.h"
#include "reduce_by_bucket.h"
#include "distribution.h"
#include "constant.h"
#include "tuple.h"
#include "stride_iterator.h"
#include "time.h"
#include "random.h"
#include "real.h"
#include "zip_iterator.h"
#include "reduce_by_key.h"
#include "iterator_equal.h"
#include "vector.h"
#include "assert.h"
#include "iterator.h"
#include "partition.h"
#include "transform_if.h"
#include "remove.h"
#include "scatter.h"
#include "bucket_indexing.h"
#include "list.h"
#include "generate.h"
#include "gather.h"
#include "transform.h"
#include "permutation_iterator.h"
#include "sort_by_key.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
