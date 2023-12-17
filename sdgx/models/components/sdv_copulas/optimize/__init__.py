"""Copulas optimization functions."""

import numpy as np


def bisect(f, xmin, xmax, tol=1e-8, maxiter=50):
    """Bisection method for finding roots.

    This method implements a simple vectorized routine for identifying
    the root (of a monotonically increasing function) given a bracketing
    interval.

    Arguments:
        f (Callable):
            A function which takes as input a vector x and returns a
            vector with the same number of dimensions.
        xmin (np.ndarray):
            The minimum value for x such that f(x) <= 0.
        xmax (np.ndarray):
            The maximum value for x such that f(x) >= 0.

    Returns:
        numpy.ndarray:
            The value of x such that f(x) is close to 0.
    """
    assert (f(xmin) <= 0.0).all()
    assert (f(xmax) >= 0.0).all()

    for _ in range(maxiter):
        guess = (xmin + xmax) / 2.0
        fguess = f(guess)
        xmin[fguess <= 0] = guess[fguess <= 0]
        xmax[fguess >= 0] = guess[fguess >= 0]
        if (xmax - xmin).max() < tol:
            break

    return (xmin + xmax) / 2.0


def chandrupatla(f, xmin, xmax, eps_m=None, eps_a=None, maxiter=50):
    """Chandrupatla's algorithm.

    This is adapted from [1] which implements Chandrupatla's algorithm [2]
    which starts from a bracketing interval and, conditionally, swaps between
    bisection and inverse quadratic interpolation.

    [1] https://github.com/scipy/scipy/issues/7242#issuecomment-290548427
    [2] https://books.google.com/books?id=cC-8BAAAQBAJ&pg=PA95

    Arguments:
        f (Callable):
            A function which takes as input a vector x and returns a
            vector with the same number of dimensions.
        xmin (np.ndarray):
            The minimum value for x such that f(x) <= 0.
        xmax (np.ndarray):
            The maximum value for x such that f(x) >= 0.

    Returns:
        numpy.ndarray:
            The value of x such that f(x) is close to 0.
    """
    # Initialization
    a = xmax
    b = xmin
    fa = f(a)
    fb = f(b)

    # Make sure we know the size of the result
    shape = np.shape(fa)
    assert shape == np.shape(fb)

    fc = fa
    c = a

    # Make sure we are bracketing a root in each case
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = np.zeros(shape, dtype=bool)

    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2 * eps

    iterations = 0
    terminate = False

    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = np.clip(a + t * (b - a), xmin, xmax)
        ft = f(xt)

        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = np.sign(ft) == np.sign(fa)
        c = np.choose(samesign, [b, a])
        b = np.choose(samesign, [a, b])
        fc = np.choose(samesign, [fb, fa])
        fb = np.choose(samesign, [fa, fb])
        a = xt
        fa = ft

        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b, a])
        fm = np.choose(fa_is_smaller, [fb, fa])

        tol = 2 * eps_m * np.abs(xm) + eps_a
        tlim = tol / np.abs(b - c)
        terminate = np.logical_or(terminate, np.logical_or(fm == 0, tlim > 0.5))

        if np.all(terminate):
            break
        iterations += 1 - terminate

        # Figure out values xi and phi
        # to determine which method we should use next
        xi = (a - b) / (c - b)
        phi = (fa - fb) / (fc - fb)
        iqi = np.logical_and(phi**2 < xi, (1 - phi) ** 2 < 1 - xi)

        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                eq1 = fa / (fb - fa) * fc / (fb - fc)
                eq2 = (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
                t = eq1 + eq2
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = np.full(shape, 0.5)
            a2, b2, c2, fa2, fb2, fc2 = a[iqi], b[iqi], c[iqi], fa[iqi], fb[iqi], fc[iqi]
            t[iqi] = fa2 / (fb2 - fa2) * fc2 / (fb2 - fc2) + (c2 - a2) / (b2 - a2) * fa2 / (
                fc2 - fa2
            ) * fb2 / (fc2 - fb2)

        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1 - tlim, np.maximum(tlim, t))

    # done!
    return xm
