## REFERENCE
# https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique

## ART Equation
# x^(k+1) = x^k + lambda * AT(b - A(x))/ATA

##
import numpy as np
import matplotlib.pyplot as plt


def ART(A, AT, b, x, mu=1e0, niter=1e2, bpos=True):

    ATA = AT(A(np.ones_like(x)))

    for i in range(int(niter)):

        x = x + np.divide(mu * AT(b - A(x)), ATA)

        if bpos:
            x[x < 0] = 0

        plt.imshow(x, cmap='gray')
        plt.title("%d / %d" % (i + 1, niter))
        plt.pause(1)
        plt.close()

    return x