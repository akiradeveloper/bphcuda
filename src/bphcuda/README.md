BPH algorithm
is a iteration of move and relax
operations.

Move can be implemented
using transform function
and the code is different 
in different environments.
In distributed environment,
particles may mvoe across the nodes
and the boudary conditions 
depends on the applications.

Relax operation on the other hand,
is pure.
It is given a set of `n_particle` particles
in `n_cell` number of cells
and just relax them.
Relax operation is 
the most complicated part of
BPH algorithm is why I 
provides this operation only
by 
`bph(n_particle, ..., n_cell, ...)` function
which is the only public API 
of this library.

See also,
performance/*.cu.
They are working examples
that uses bphcuda library.
