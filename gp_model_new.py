from celerite2.jax import terms, GaussianProcess
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

def harmonic_sho_model(t, y, yerr, yquarters, f0, psd_freq=None, predict_flux=False):
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
    psd_freq : array_like, optional
        If given, each sample will record the GP PSD at these frequencies (per
        cycle, not per radian).
    predict_flux : bool, default=False
        If given, each sample will record the model's estimate of the expected
        flux at the observation times.
    """
    uquarters, quarter_indices = np.unique(yquarters, return_inverse=True)

    n_in_quarters = np.bincount(quarter_indices)
    mu_quarters = np.bincount(quarter_indices, weights=y) / n_in_quarters
    var_quarters = np.bincount(quarter_indices, weights=np.square(y - mu_quarters[quarter_indices])) / n_in_quarters
    std_quarters = np.sqrt(var_quarters)

    rel_std_quarters = std_quarters / mu_quarters

    T = np.max(t) - np.min(t)

    coords = {'quarters': uquarters}
    if psd_freq is not None:
        coords['frequencies'] = psd_freq
    if predict_flux:
        coords['times'] = t

    nquarters = uquarters.shape[0]

    # Want a prior on the log_scale_factors that is N(0, 1/sqrt(n)) (very
    # broad); expect a posterior that is N(0, rel_std_quarters / sqrt(n)).
    # Let log_scale_factors = rel_std_quarters / sqrt(n) *
    # log_scale_factors_scaled so that the posterior on
    # log_scale_factors_scaled is N(0,1)-ish.  Then N(0, 1/sqrt(n)) on
    # log_scale_factors produces a prior on log_scale_factors_scaled that is
    # N(0, 1/rel_std_quarters).
    log_scale_factors_scaled = numpyro.sample('log_scale_factors_scaled', dist.Normal(0, 1/rel_std_quarters), sample_shape=nquarters)
    log_scale_factors = rel_std_quarters / np.sqrt(n_in_quarters) * log_scale_factors_scaled
    scale_factors = numpyro.deterministic('scale_factors', jnp.exp(log_scale_factors))

    # mus is the mean flux, and also the scaling factor for each quarter's
    # GP.  That is, we are fitting flux = mus*(1+gp).
    mus = numpyro.deterministic('mus', mu_quarters*scale_factors)

    y_scaled = y / mus[quarter_indices]
    y_centered = y_scaled - 1.0

    y_err_scaled = yerr / mus[quarter_indices]

    # log_err_scale = pm.Uniform('log_err_scale', -np.log(2), np.log(2))
    # err_scale = pm.Deterministic('err_scale', pt.exp(log_err_scale))

    # log_period_scaled = pm.Normal('log_period_scaled', 0, 1)
    # log_period = pm.Deterministic('log_period', -pt.log(f0) + f_frac_uncert*log_period_scaled)
    log_period = numpyro.sample('log_period', dist.Uniform(-jnp.log(f0) - np.log(np.sqrt(2)), -jnp.log(f0) + np.log(np.sqrt(2))))
    period = numpyro.deterministic('period', jnp.exp(log_period))
    _ = numpyro.deterministic('f0', 1/period)

    # We want to impose a very broad prior, HN(0.1) on the sigma parameter
    # (i.e. up to 10% variability), but we expect a posterior that is
    # ~rel_std_quarters wide, so we define sigma_scaled = sigma /
    # np.median(rel_std_quarters) so that sigma_scaled is unit-scale
    # posterior.  Then a HN(0.1) on sigma induces a HN(0.1 /
    # np.median(rel_std_quarters)) prior on sigma_scaled.
    sigma_scaled = numpyro.sample('sigma_scaled', dist.HalfNormal(0.1 / np.median(rel_std_quarters)))
    sigma = numpyro.deterministic('sigma', sigma_scaled * np.median(rel_std_quarters))

    # logfrac = pm.Uniform('logfrac', np.log(0.1), np.log(10))
    # frac = pm.Deterministic('frac', pt.exp(logfrac))
    frac = numpyro.sample('frac', dist.Uniform(0, 1))

    dQ1 = numpyro.sample('dQ1', dist.LogNormal(np.log(5), 1))
    dQ0 = numpyro.sample('dQb', dist.LogNormal(np.log(5), 1))
    Q0 = numpyro.deterministic('Q0', 0.5 + dQ1 + dQ0)
    Q1 = numpyro.deterministic('Q1', 0.5 + dQ1)

    kernel1 = terms.RotationTerm(sigma=sigma, period=period, Q0=dQ1, dQ=dQ0, f=frac)

    longest_period = 1.0 / (f0 / np.sqrt(2))  
    
    # For RealTerm: c parameter (decay rate) controls 1/c = characteristic timescale
    # We want the longest allowed timescale to be the longest period, so c_max = 1/longest_period
    # We set c_min = 1/T (where T is total observation time) to avoid very long timescales
    T_obs = np.max(t) - np.min(t)  # Total observation time
    c_min = 1.0 / T_obs  # Longest allowed timescale is the total observation time
    c_max = 1.0 / longest_period  # Shortest allowed timescale is the longest rotation period
    
    # Use a log-uniform prior for c between these bounds
    log_c = numpyro.sample('log_c', dist.Uniform(np.log(c_min), np.log(c_max)))
    c = numpyro.deterministic('c', jnp.exp(log_c))
    
    # For the amplitude parameter 'a', we want sigma_red_noise to follow the same
    # pattern as the main sigma parameter for the rotation kernel
    sigma_red_noise_scaled = numpyro.sample('sigma_red_noise_scaled', dist.HalfNormal(0.1 / np.median(rel_std_quarters)))
    sigma_red_noise = numpyro.deterministic('sigma_red_noise', sigma_red_noise_scaled * np.median(rel_std_quarters))
    
    # The variance of the red noise process is just 'a'
    a_red_noise = numpyro.deterministic('a_red_noise', sigma_red_noise * sigma_red_noise)

    kernel2 = terms.RealTerm(a=a_red_noise, c=c)

    kernel = kernel1 + kernel2

    gp = GaussianProcess(kernel)
    gp.compute(t, yerr=y_err_scaled, quiet=True)

    # The GP will compute p(y_centered | parameters), but we need p(y |
    # parameters), so require a log-Jacobian term that is
    # log-det(d(y_centered)/dy), or -sum(log(mus))
    numpyro.factor('log_likelihood', gp.log_likelihood(y_centered))
    numpyro.factor('log_likelihood_jacobian', -pt.sum(pt.log(mus[quarter_indices])))

    if predict_flux:
        numpyro.deterministic('gp_mean_model', mus[quarter_indices]*(1 + gp.predict(y_centered, t=t, return_var=False)))

    if psd_freq is not None:
        psd = gp.kernel.get_psd(2*np.pi*psd_freq)
        numpyro.deterministic('psd', psd*2*np.pi, dims=['frequencies']) # Convert from per-radian to per-cycle