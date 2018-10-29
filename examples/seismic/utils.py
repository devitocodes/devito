from scipy import ndimage

__all__ = ['scipy_smooth']


def scipy_smooth(img, sigma=5):
    """
    Smooth the input with scipy ndimage utility
    """
    return ndimage.gaussian_filter(img, sigma=sigma)
