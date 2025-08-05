from celerite2.pymc import terms, GaussianProcess
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def harmonic_sho_model(t, y, yerr, yquarters, f0, harmonics, mu_mu, mu_sigma, sho_sigma_prior, f_frac_uncert=0.1, psd_freq=None, predict_times=None):
    """A (quasi)harmonic simple-harmonic-oscillator GP model for a time series.

    Produce a pymc model for the given time multi-quarter / multi-period time
    series that represents it as a celerite GP with a sum of SHO terms.  The
    frequencies of the SHO terms are encouranged by a prior to be harmonics of
    the given fundamental frequency, as might be expected for the quasi-periodic
    oscillations due to the rotation of a spotty star.  The model contains a
    per-quarter constant flux offset (i.e. per-quarter mean term) to account for
    a varying zero-point period-by-period.

    Parameters
    ----------

    t : array_like
        The time values of the time series.
    y : array_like
        The flux values of the time series.
    yerr : array_like
        The uncertainties on the flux values.
    yquarters : array_like
        The period number of each time point (these need not be contiguous if
        there are periods in which the target was not observed).
    f0 : float
        A guess at the fundamental frequency of the oscillators (i.e. the
        inverse of the estimated rotation period).  Each harmonic will have a
        LogNormal prior for its frequency peaking at `i*f0` for harmonic `i`.
    harmonics : int array_like
        The harmonics to include in the model (the first element should be `1`,
        followed by whatever multiples of the fundamental frequency are
        desired).
    mu_mu : array_like
        The mean of the Normal prior applied to the per-period flux offsets.
    mu_sigma : float
        The standard deviation of the Normal prior applied to the per-period
        flux offsets.
    sho_sigma_prior : float
        The peak of the LogNormal prior applied to the RMS variability of the
        SHO terms; eac SHO term's `sigma` parameter will have a LogNormal
        distribution peaking at this value and with a width that gives a prior
        two-sigma span that is a factor of 10 smaller to a factor of 10 larger
        than this value.
    f_frac_uncert : float, default=0.1
        The standard deviation of the log-frequency prior for the harmonics.
    psd_freq : array_like, optional
        If given, each sample will record the GP PSD at these frequencies (per
        cycle, not per radian).
    predict_times : array_like, optional
        If given, each sample will record the model's estimate of the expected
        flux at these times.
    """
    uquarters, quarter_indices = np.unique(yquarters, return_inverse=True)

    nharmonics = len(harmonics)

    coords = {'harmonics': harmonics, 'quarters': uquarters, 'times': t}
    if psd_freq is not None:
        coords['frequencies'] = psd_freq
    if predict_times is not None:
        coords['predict_times'] = predict_times

    with pm.Model(coords=coords) as model:
        nquarters = mu_mu.shape[0]

        mus_scaled = pm.Normal('mus_scaled', 0, 1, shape=(nquarters,), dims=['quarters'])
        mus = pm.Deterministic('mus', mus_scaled * mu_sigma + mu_mu, dims=['quarters'])

        y_centered = y - mus[quarter_indices]

        log_fs_scaled = pm.Normal('log_fs_scaled', 0, 1, shape=(nharmonics,), dims=['harmonics'])
        log_fs = pm.Deterministic('log_fs', log_fs_scaled*f_frac_uncert + pt.log(f0) + pt.log(harmonics), dims=['harmonics'])
        fs = pm.Deterministic('fs', pt.exp(log_fs), dims=['harmonics'])

        log_Qs_scaled = pm.Normal('log_Qs_scaled', 0, 1, shape=(nharmonics,), dims=['harmonics'])
        log_Qs = pm.Deterministic('log_Qs', log_Qs_scaled * pt.log(10)/3 + pt.log(10), dims=['harmonics']) # LogNormal prior, peaks at Q = 10, 3-sigma width is a factor of 10
        Qs = pm.Deterministic('Qs', pt.exp(log_Qs), dims=['harmonics'])

        log_sigma_scaled = pm.Normal('log_sigma_scaled', 0, 1, shape=(nharmonics,), dims=['harmonics'])
        log_sigma = pm.Deterministic('log_sigma', pt.log(sho_sigma_prior) + pt.log(10)/2*log_sigma_scaled, dims=['harmonics'])
        sigmas = pm.Deterministic('sigmas', pt.exp(log_sigma), dims=['harmonics'])

        trms = [terms.SHOTerm(w0=2*np.pi*fs[i], Q=Qs[i], sigma=sigmas[i]) for i in range(nharmonics)]
        kernel = terms.TermSum(*trms)

        gp = GaussianProcess(kernel)
        gp.compute(t, yerr=yerr, quiet=True)
        pm.Potential('log_likelihood', gp.log_likelihood(y_centered))

        if predict_times is not None:
            pm.Deterministic('gp_mean_model', gp.predict(y_centered, t=predict_times, return_var=False) + mus[quarter_indices], dims=['predict_times'])

        if psd_freq is not None:
            psd = gp.kernel.get_psd(2*np.pi*psd_freq)
            pm.Deterministic('psd', psd*2*np.pi, dims=['frequencies']) # Convert from per-radian to per-cycle 

        return model