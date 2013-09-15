Thrusting library is
a set of complimentary functionality for
Thrust library.

Some of the helpers are just wrapper 
for the original that changes the API
slightly.

The original API design is not 
for real and complicated applications like bphcuda.

I re-defined the API in such a way that
`void transform(begin, end, out, f)`
to
`void transform(len, begin, end, f)`.

The trivial tests for those wrappers are also important
to make sure that the routines 
work as we expect.

First, `rake build` to 
Generate all the helper functions.
