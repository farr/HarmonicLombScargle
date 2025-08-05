import numpy as np
from tqdm import tqdm

class HarmonicLombScargle(object):
    def __init__(self, t, flux, quarters, nharmonics):
        self._t = np.array(t)
        self._flux = np.array(flux)
        self._quarters = np.array(quarters)
        self._nharmonics = nharmonics

        uquarters = np.unique(quarters)
        bins = np.concatenate((uquarters, [uquarters[-1] + 1]))

        self._nquarters = len(uquarters)
        self._quarter_indices = np.digitize(quarters, bins) - 1

    @property
    def t(self):
        return self._t
    
    @property
    def flux(self):
        return self._flux
    
    @property
    def quarters(self):
        return self._quarters
    
    @property
    def nharmonics(self):
        return self._nharmonics
    
    @property
    def quarter_indices(self):
        return self._quarter_indices
    
    @property
    def nquarters(self):
        return self._nquarters
    
    def _design_matrix(self, f):
        nt = self.t.shape[0]
        nq = self.nquarters
        nh = self.nharmonics

        M = np.zeros((nt, nq + 2*self.nharmonics))

        M[np.arange(nt), self.quarter_indices] = 1
        for j in range(nh):
            M[:, nq + 2*j] = np.sin(2*np.pi*(j+1)*f*self.t)
            M[:, nq + 2*j + 1] = np.cos(2*np.pi*(j+1)*f*self.t)

        return M
    
    def _best_fit_and_logl(self, f):
        M = self._design_matrix(f)
        x, (rss,), _, _ = np.linalg.lstsq(M, self.flux)

        return x, -0.5*rss, M
    
    def relative_frequency_grid(self, fractional_f_spacing):
        fmin = 1/(self.t[-1] - self.t[0])
        fmax = 0.5*np.median(1/np.diff(self.t))

        return np.logspace(np.log10(fmin), np.log10(fmax), int(np.log10(fmax/fmin)/np.log10(1 + fractional_f_spacing)) + 1)
    
    def fine_frequency_grid(self, f0, n=128):
        df = 1/(self.t[-1] - self.t[0])
        fmin = max(0, f0 - 2*df)
        fmax = f0 + 2*df
        
        return np.linspace(fmin, fmax, n)

    def logl_on_grid(self, fs, progress=True):
        if progress:
            pm = tqdm
        else:
            pm = lambda x: x

        return np.array([self._best_fit_and_logl(f)[1] for f in pm(fs)])
    
    def interpolated_best_frequency(self, fs, logls):
        imax = np.argmax(logls)
        if imax == 0 or imax == len(fs) - 1:
            return fs[imax]
        
        f0, f1, f2 = fs[imax - 1:imax + 2]
        l0, l1, l2 = logls[imax - 1:imax + 2]

        return (f2*f2*(l0-l1) + f0*f0*(l1-l2) + f1*f1*(l2-l0)) / (2*(f2*(l0-l1) + f0*(l1-l2) + f1*(l2-l0)))

    def best_fit_params(self, f):
        return self._best_fit_and_logl(f)[0]
    
    def predict_flux(self, f):
        bfp, _, M = self._best_fit_and_logl(f)
        return np.dot(M, bfp)

    def periodic_lightcurve(self, ts, f, bfp):
        y = np.zeros_like(ts)

        for i in range(self.nharmonics):
            y += bfp[self.nquarters + 2*i]*np.sin(2*np.pi*(i+1)*f*ts)
            y += bfp[self.nquarters + 2*i + 1]*np.cos(2*np.pi*(i+1)*f*ts)
    
        return y