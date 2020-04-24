
#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------


# ---------------------------------------------------
# Check convergence for classifier for early stopping
# ---------------------------------------------------
def check_convergence(
        prev_loss,
        cur_loss,
        cur_step,
        iter_below_tol,
        abs_loss_chg_tol = 0.001,
        min_num_iter = 100,
        max_iter_below_tol = 20
):
    """Checks if training for a model has converged."""
    has_converged = False

    # Check if we have reached the desired loss tolerance.
    loss_diff = abs(prev_loss - cur_loss)
    if loss_diff < abs_loss_chg_tol:
        iter_below_tol += 1
    else:
        iter_below_tol = 0

    if iter_below_tol >= max_iter_below_tol:
        has_converged = True

    if cur_step < min_num_iter:
        has_converged = False

    return has_converged, iter_below_tol

