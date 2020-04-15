def setup_wOverQ(wOverQ, w, qmin, qmax, npad, sigma=None):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model.

    Parameters
    ----------
    wOverQ : Function
        The omega over Q field used to implement attenuation in the model, 
        and the absorbing boundary condition for outgoing waves.
    w : float32
        center angular frequency, e.g. peak frequency of Ricker source wavelet 
        used for modeling.
    qmin : float32
        Q value at the edge of the model. Typically set to 0.1 to strongly 
        attenuate outgoing waves.
    qmax : float32
        Q value in the interior of the model. Typically set to 100 as a 
        reasonable and physically meaningful Q value.
    npad : int
        Number of points in the absorbing boundary region. Note that we expect 
        this to be the same on all sides of the model.
    sigma : float32
        sigma value for call to scipy gaussian smoother, default 5.
    """
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)

    sigma = sigma or npad//5
    
    t1 = timer()
    lqmin = np.log(qmin);
    lqmax = np.log(qmax);

    kxMin,kzMin = np.ogrid[:nxpad,:nzpad]
    kxArr,kzArr = np.minimum(kxMin,nxpad-1-kxMin),np.minimum(kzMin,nzpad-1-kzMin)
    nval1 = np.minimum(kxArr,kzArr)
    nval2 = np.minimum(1,nval1/(npad))
    nval3 = np.exp(lqmin+nval2*(lqmax-lqmin))
    wOverQ.data[:,:] = w / nval3

    if sigma > 0:
        wOverQ.data[:] = gaussian_smooth(wOverQ.data, sigma=sigma)

    # report min/max output Q value
    q1 = (np.min(1 / (wOverQ.data / w)))
    q2 = (np.max(1 / (wOverQ.data / w)))
    t2 = timer()
    print("setup_wOverQ ran in %.4f seconds -- min/max Q values; %8.4f %8.4f" % (t2-t1, q1, q2)) 
