"""falsesync — aggregation-induced false synchrony diagnostics.

Core model: Y_i(t) = a_i + b_i(t) + A_i g_i((t - tau_i)/h_i) + eps_i(t).
The aggregate breakpoint is an operator-dependent functional
T_M(F_tau, g, w, ...) which in general equals no moment of F_tau.
See math_specification.md and proofs/ for P1-P8.
"""

__version__ = "0.1.1"
