from celerite2.pymc import terms, GaussianProcess
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def harmonic_sho_model(t, y, yerr, yquarters, f0, f_frac_uncert, mu_mu, mu_sigma, sho_sigma_prior, psd_freq=None, predict_flux=False):
    """A (quasi)harmonic simple-harmonic-oscillator GP model for a time series.

    Produce a pymc model for the given time multi-quarter / multi-period time
    series that represents it as a celerite GP with a sum of two SHO terms
    designed to find the fundamental and first harmonic of rotation (i.e. a
    celerite `RotationTerm`).  The fundamnetal frequency is given a log-normal
    prior centered at `f0` with width `f_frac_uncert`.  The model contains a
    per-quarter constant flux offset (i.e. per-quarter mean term) to account for
    a varying zero-point period-by-period, as well as a red-noise, "real"
    celerite term to account for additional variability not captured by the
    SHOs.  The real term is also known as a "damped random walk," and its "knee"
    frequency is constrained to be below the fundamental frequency of the SHOs.

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
    f_frac_uncert : float
        The standard deviation of the log-frequency prior for the harmonics.
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
    psd_freq : array_like, optional
        If given, each sample will record the GP PSD at these frequencies (per
        cycle, not per radian).
    predict_flux : bool, default=False
        If given, each sample will record the model's estimate of the expected
        flux at the observation times.
    """
    uquarters, quarter_indices = np.unique(yquarters, return_inverse=True)

    T = np.max(t) - np.min(t)
    fmin = 1/T
    fmax = 1/(2*np.min(np.diff(t)))

    coords = {'quarters': uquarters}
    if psd_freq is not None:
        coords['frequencies'] = psd_freq
    if predict_flux:
        coords['times'] = t

    with pm.Model(coords=coords) as model:
        nquarters = mu_mu.shape[0]

        mus_scaled = pm.Normal('mus_scaled', 0, 1, shape=(nquarters,), dims=['quarters'])
        mus = pm.Deterministic('mus', mus_scaled * mu_sigma + mu_mu, dims=['quarters'])

        y_centered = y - mus[quarter_indices]

        log_err_scale = pm.Uniform('log_err_scale', -np.log(2), np.log(2))
        err_scale = pm.Deterministic('err_scale', pt.exp(log_err_scale))

        log_period_scaled = pm.Normal('log_period_scaled', 0, 1)
        log_period = pm.Deterministic('log_period', -pt.log(f0) + f_frac_uncert*log_period_scaled)
        period = pm.Deterministic('period', pt.exp(log_period))
        _ = pm.Deterministic('f0', 1/period)

        log_sigma_scaled = pm.Normal('log_sigma_scaled', 0, 1)
        log_sigma = pm.Deterministic('log_sigma', pt.log(sho_sigma_prior) + pt.log(10)/2*log_sigma_scaled)
        sigma = pm.Deterministic('sigma', pt.exp(log_sigma))

        frac = pm.Uniform('frac', 0, 1)

        dQ1 = pm.LogNormal('dQ1', pt.log(5), 1)
        dQ0 = pm.LogNormal('dQb', pt.log(5), 1)
        Q0 = pm.Deterministic('Q0', 0.5 + dQ1 + dQ0)
        Q1 = pm.Deterministic('Q1', 0.5 + dQ1)

        kernel = terms.RotationTerm(sigma=sigma, period=period, Q0=dQ1, dQ=dQ0, f=frac)

        gp = GaussianProcess(kernel)
        gp.compute(t, yerr=yerr*err_scale, quiet=True)
        pm.Potential('log_likelihood', gp.log_likelihood(y_centered))

        if predict_flux:
            pm.Deterministic('gp_mean_model', gp.predict(y_centered, t=t, return_var=False) + mus[quarter_indices], dims=['times'])

        if psd_freq is not None:
            psd = gp.kernel.get_psd(2*np.pi*psd_freq)
            pm.Deterministic('psd', psd*2*np.pi, dims=['frequencies']) # Convert from per-radian to per-cycle 

        return model