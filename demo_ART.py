## REFERENCE
# https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique

## ART Equation
# x^(k+1) = x^k + lambda * AT(b - A(x))/ATA

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from scipy.io import loadmat
from scipy.stats import poisson
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
# from skimage.metrics import mean_squared_error as mse
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
from ART import ART

## SYSTEM SETTING
N = 512
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW + 1)
THETA = THETA[:-1]

A = lambda x: radon(x, THETA, circle=False).astype(np.float32)
AT = lambda y: iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)/(np.pi/(2*len(THETA)))
AINV = lambda y: iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

## DATA GENERATION
x = loadmat('XCAT512.mat')['XCAT512']
p = A(x)
x_full = AINV(p)

## LOW-DOSE SINOGRAM GENERATION
i0 = 5e4
pn = np.exp(-p)
pn = i0*pn
pn = poisson.rvs(pn)
pn[pn < 1] = 1
pn = -np.log(pn/i0)
pn[pn < 0] = 0

y = pn

## Algebraic Reconstruction Technique (ART) INITIALIZATION
x_low = AINV(y)
x0 = np.zeros_like(x)
mu = 1e0
niter = 2e2
bpos = True

x_art = ART(A, AT, y, x0, mu, niter, bpos)

## CALCULATE QUANTIFICATION FACTOR
x_low[x_low < 0] = 0
x_art[x_art < 0] = 0
nor = np.amax(x)

mse_x_low = mse(x/nor, x_low/nor)
mse_x_art = mse(x/nor, x_art/nor)

psnr_x_low = psnr(x/nor, x_low/nor)
psnr_x_art = psnr(x/nor, x_art/nor)

ssim_x_low = ssim(x_low/nor, x/nor)
ssim_x_art = ssim(x_art/nor, x/nor)

## DISPLAY
wndImg = [0, 0.03]
wndPrj = [0, 6]

plt.subplot(241)
plt.imshow(x, cmap='gray', vmin=wndImg[0], vmax=wndImg[1])
plt.axis('off')
plt.axis('image')
plt.title('Ground truth')

plt.subplot(242)
plt.imshow(x_full, cmap='gray', vmin=wndImg[0], vmax=wndImg[1])
plt.axis('off')
plt.axis('image')
plt.title('full-dose')

plt.subplot(243)
plt.imshow(x_low, cmap='gray', vmin=wndImg[0], vmax=wndImg[1])
plt.axis('off')
plt.axis('image')
plt.title('low-dose\nMSE: %.4f\nPSNR: %.4f\nSSIM: %.4f' % (mse_x_low, psnr_x_low, ssim_x_low))

plt.subplot(244)
plt.imshow(x_art, cmap='gray', vmin=wndImg[0], vmax=wndImg[1])
plt.axis('off')
plt.axis('image')
plt.title('ART\nMSE: %.4f\nPSNR: %.4f\nSSIM: %.4f' % (mse_x_art, psnr_x_art, ssim_x_art))

plt.subplot(246)
plt.imshow(p, cmap='gray', vmin=wndPrj[0], vmax=wndPrj[1])
plt.title('full-dose\n(VIEW: %d)' % VIEW)
plt.xlabel('View')
plt.ylabel('Detector')

plt.subplot(247)
plt.imshow(y, cmap='gray', vmin=wndPrj[0], vmax=wndPrj[1])
plt.title('low-dose\n(VIEW: %d)' % VIEW)
plt.xlabel('View')
plt.ylabel('Detector')

plt.subplot(248)
plt.imshow(y - p, cmap='gray')
plt.title('full-dose - low-dose\n(Poisson noise)')
plt.xlabel('View')
plt.ylabel('Detector')

plt.show()
