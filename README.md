SpiNNaker_PDP2 README
=====================

-----------------------------------------------
***********************************************
-----------------------------------------------
GIT tag: final.pacman.48

This is the FINAL commit for the pacman48-based
version of SpiNNaker_PDP2.

This code has been ported to work based on the
SpiNNaker Graph Front End platform. Any further
development will be on this platform.

Most of the issues reported in this file have
now been added as issues in the GIT repository
and will, therefore, de removed from this file.
-----------------------------------------------
***********************************************
-----------------------------------------------

This file lists bugs and issues related to the development of the
SpiNNaker C-code to implement multi-layer perceptrons in lens
style. There is no particular order or structure to the file.

Fixed-point representation
--------------------------
1. Weights have now been changed to s16.15 representation, which allows
for much greater precision, and also for the handling of much larger
weights.  Previously, with the s3.12 representation, weights in some
examples exceeded the [-7.0, 7.0) range.  Hopefully this is no longer
an issue.

2. With large weights, partial nets (s4.27 representation) can get
outside the [-16.0, 16.0) range. May need to use a longer type and
saturate. This may also be the case for error deltas in backprop.


Stopping criteria
-----------------
1. pacman uses incorrect fixed-point representation for stopping
criteria. Currently fixed in the places where they are used, should
fix it in pacman or during initialization to avoid potential
inconsistencies.

2. Add SpiNNaker support for grace period. It shouldn't even evaluate
errors during that period.


Output files have differences between lens and SpiNNaker
--------------------------------------------------------
1. Outputs reported are inconsistent in tick -1. lens reports 0 while
SpiNNaker reports 0.5. lens itself is not consistent (output is 0 in
rand10x40 and 0.5 in rogers - may have to do with output integrator in
rand10x40).

2. lens reports the number of weight updates while SpiNNaker doesn't.

3. lens reprots the actual number of ticks fro every example while
SpiNNaker reports the maximum.

4. target values are not reported by lens during the grace period
while SpiNNaker does it every tick.

PACMAN Changes
--------------
In order to change weight representations from s3.12 to s16.15,
changes to pacman were required.  Therefore the correct pacman code is
required to be able to run this branch of PDP2.

Consequences
------------
The changes to pacman for the weight representation had the fortunate
side effect of changing inputs and targets from 16-bit to 32-bit
numbers, meaning that a value of 1 can now be properly represented
(instead of 0.999969, as previously.  This required the addition of 2
minor "hacks" to make things consistent:

1. The initOutput value for the bias units is still 0.999969.  This
has therefore been corrected within the code at the point at which
initial outputs are loaded into `t_outputs`, as I have been unable to
ascertain the value of 0.999969 is being picked up from.

2. Targets are now 32 bit values, rather than 16 bit values, but
`output_mon_lens` expects them to be loaded into the array `my_data`
as 16 bit values.  When cast to 16 bit values, a target of 1 becomes
-1 because the 16 bit activation is s0.15.  This was not previously a
problem because a true 1 was never actually represented, 0.999969 was
used instead.  Therefore the code at this point now checks whether the
target is 1, and if so, loads the value `SPINN_SHORT_ACTIV_MAX`
(0.999969) into the array.  `output_mon_lens` is then able to handle
this as before, and outputs a target value of 1.
