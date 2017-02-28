SpiNNaker_PDP2 README
=====================

This file lists bugs and issues related to the development of the
SpiNNaker C-code to implement multi-layer perceptrons in lens
style. There is no particular order or structure to the file.


Fixed-point representation
--------------------------
1. Weights (s3.12 representation) are outside the current [-7.0, 7.0]
range in some examples. Check if we need to extend the range and use a
different representation.
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
