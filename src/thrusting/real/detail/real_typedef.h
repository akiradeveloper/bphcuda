#pragma once

namespace thrusting {
#ifdef THRUSTING_USING_DOUBLE_FOR_REAL
    typedef double real;
#else
    typedef float real;
#endif
}
