import numpy as np
import matplotlib.pyplot as plt
def polyfit2d(z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x_vector = np.arange(z.shape[1])
    y_vector = np.arange(z.shape[0])
    x, y = np.meshgrid(x_vector,y_vector)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    soln = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
    fitted_surf = np.polynomial.polynomial.polygrid2d(x_vector, y_vector, soln[0].reshape((kx+1,ky+1)))
    fitted_surf = fitted_surf.transpose()
    #plt.matshow(fitted_surf)
    fig, axs = plt.subplots(1, 2, figsize=(20,20))
    # Plot first image
    axs[0].imshow(fitted_surf)
    axs[0].set_title('Fitted surface')
    axs[1].imshow(z-fitted_surf,vmin=-35, vmax=35)
    axs[1].set_title('Fit_surf Substracted')
    return z-fitted_surf