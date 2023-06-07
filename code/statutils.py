# Load the basics
import numpy as np
from scipy.special import kn, kv
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy import interpolate

# Import the waveform packages
from gbgpu.gbgpu import GBGPU
from gbgpu.utils.utility import get_fdot, get_N
from gbgpu.utils.constants import *
from eryn.utils import TransformContainer

# Check if we have GPUs!
try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice
    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp
    gpu_available = False

# ---------------------------- UCB Template class ----------------------- #
class ucb:
    """
    Generating the waveform of a Ultra Compact Binary source and projecting it onto the LISA arms.
    Using the GBGPU package developed by M. Katz that can be found here: https://github.com/mikekatz04/GBGPU
    
    Needs the frequency array, the set of modes, the gpu flag, the start time of the measurement,
    the end time of the measurement, and the reference frequency.
    
    """
    def __init__(self, f_lims=None, f_inds=None, gpu=False, tobs=1.0, dt=1.0, use_tdi2=False):
        
        self._template = GBGPU(use_gpu=gpu_available)
        self._inds = f_inds
        self._tdi2 = use_tdi2
        
        # Just a check in order to avoid weird errors later
        if self._inds is None or f_lims is None:
            raise TypeError("Error: Please define the frequency limits and the indices to be used for the waveform computation.")
        
        self._waveform_kwargs = dict(N=f_inds[1]-f_inds[0], 
                                     dt=dt, 
                                     T=tobs,
                                     tdi2=self._tdi2)
        
        trans = {
            0: (lambda x: 10**(x)),
            1: (lambda x: x * 1e-3),
            2: (lambda x: 10**(x)),
            5: (lambda x: np.arccos(x)),
            8: (lambda x: np.arcsin(x)),}

        self._ndim_full = 9
        fill_dict = {"fill_inds": np.array([3]), "ndim_full": self._ndim_full, "fill_values": np.array([0.0])}
        self._transform_fn = TransformContainer(
            parameter_transforms=trans, fill_dict=fill_dict)

    def eval(self, params):
        """ Evaluating the model at a given parameter set.
        Args:
            params (ndarray): The parameter values to evaluate the likelihood on.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_binaries)`` if 2D for more than one binary.
            
        Returns:
            ndarray: A, E, T channels of the waveform. 
        """

        params = np.insert(params, 3, 0.0, axis=1) # inject with zeros for the fddot (we do not want to sample for it?)
        # Define the injection parameters                                            
        injct = self._transform_fn.transform_base_parameters(params, 
                        return_transpose=False).reshape(-1, self._ndim_full).T
        self._template.run_wave(*injct, **self._waveform_kwargs,) # inject into the template function
        A_temp, E_temp = self._template.A, self._template.E # Get the A and E channels
        
        return np.atleast_2d(A_temp), np.atleast_2d(E_temp), None # We need 3 outputs for the likelihood    
    
#------------------------------------- Noise model definition ----------------------------- #
clight = 299792458.0
lisaLT = 2.5e9/clight

class noise_model:
    """
      Class for the nosie model. We only need the f-vector to evaluate it
    """
    def __init__(self, fvec):
        self._f = xp.array(fvec) # Ensure we get cuda array
        
    def eval(self, params):
        """LISA noise model for AE channels. The parameters are the log-base-10 of the acceleration and 
            displacement noises.
        """
        params = xp.array(params) # <---- Ensure it's a numpy or cupy array
        # Be careful with the dimensions here
        if params.ndim == 1:
            Sa, Si = params[:,None]
        else:
            Sa, Si = params[:,0], params[:,1]
        x   = 2.0 * xp.pi * lisaLT * self._f
        Spm = 10**(Sa[:,None]) * (1.0 + (0.4e-3/self._f)**2) * (1.0+(self._f/8e-3)**4) * (2.0*xp.pi*self._f)**(-4.) * (2.0*xp.pi*self._f/clight)**2 
        Sop = 10**(Si[:,None]) * (1.0 + (2.e-3/self._f)**4) * (2.0 * xp.pi * self._f/clight)**2
        S0  = 8.0 * xp.sin(x)**2 * (2.0 * Spm * (3.0 + 2.0*xp.cos(x) + xp.cos(2.*x)) + Sop * (2.0 + xp.cos(x)))
        return xp.array(S0)
    def __call__(self, p):
        return self.eval(p)
    
    
# ---------------------------- Log-likelihood class ----------------------- #
class loglike:
    """
      Classes for the different log-likelihood functions 
      
      [Need to write more here]
      
    """
    def __init__(self, data, template, f, df, Sn, gpu=False, infs=-1e300, x_knots=None):
        self._A   = xp.array(data[0])
        self._E   = xp.array(data[1])
        self._Sn  = Sn        # The PSD of the noise: Assumed array or function
        self._df  = df
        self._d   = len(data) 
        self._lam = (self._d + 1)/2
        self._f   = f
        self._C0  = ((1 - self._d)/2)*np.log(2.*np.pi)
        self._Nd  = self._f.shape[0] # Number of data points.
        self._template = template
        self._use_gpu = gpu
        self._inf = infs
        self._logfreq = np.log10(np.array(self._f))
        self._x_knots = x_knots
        if x_knots is not None:
            self._interpdim = len(x_knots)
            self._x_knots = np.log10(x_knots)
        else:
            self._interpdim = None
        
    # For the case of GPU use, we need to output the numpy arrays
    def _prepare_outputs(self, like):
        """ Utility function to export numpy from cuda arrays"""
        if self._use_gpu:
            return like.get()
        else:
            return like

    # Utility function
    def _get_yy(self, theta):
        """ Utility function to get yy"""
        Af, Ef, _ = self._template.eval(theta)         
        res_A = (self._A - Af)
        res_E = (self._E - Ef)
        LA = 2. * self._df * (res_A.conj() * res_A)
        LE = 2. * self._df * (res_E.conj() * res_E)
        return (LA + LE)

    # Gaussian log-likelihood - known noise
    def gaussian(self, theta):
        """ Gaussian log-likelihood:
        """
        L = self._prepare_outputs( - .5 * xp.sum( self._get_yy(theta) / self._Sn, axis=-1).real )
        L = np.nan_to_num(L, copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf)
        return L
    
    # Gaussian log-likelihood - Fitting the noise model
    def gaussian_noise_fit(self, theta):
        """ Gaussian log-likelihood, noise model fit:
        """
        yy = self._get_yy(theta[:,:-2])

        # Estimate the PSD of the noise
        Sn = xp.array(self._Sn(theta[:,-2:]))
        L  = self._prepare_outputs( - .5 * xp.sum( yy / Sn + 2. * xp.log(Sn), axis=-1).real )
        L = np.nan_to_num(L, copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf)
        return L
    
    # Gaussian log-likelihood - Fitting the noise model
    def gaussian_noise_fit_level(self, theta):
        """ Gaussian log-likelihood, noise level fit:
        """
        yy = self._get_yy(theta[:,:-1])

        # Estimate the PSD of the noise (Just a level here)
        Sn = xp.array(theta[:,-1])[:,None] * self._Sn
        
        L  = self._prepare_outputs( - .5 * xp.sum( yy / Sn + 2. * xp.log(Sn), axis=-1).real )
        L  = np.nan_to_num(L, copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf)
        return L

    # Hyperbolic log-likelihood
    def hyperbolic(self, theta):
        """ See Prause Eberlein eq 4.21
        """
        alpha = 10**theta[:,-2]
        delta = 10**theta[:,-1]
        Kappa = kv(self._lam, alpha*delta)
        yy = self._get_yy(theta[:,:-2])
        term_under_sqrt = xp.sum( xp.sqrt( xp.add( xp.array(delta[:,None])**2, yy / self._Sn ) ), axis=-1).real

        if self._use_gpu:
            term_under_sqrt = term_under_sqrt.get()

        L = self._Nd*(self._lam*(np.log(alpha/delta)) + self._C0 \
                    - np.log(2*alpha) - np.log(Kappa))  \
                        - alpha * term_under_sqrt
        L = np.nan_to_num(L, copy=True, nan=self._inf, posinf=self._inf, neginf=self._inf)
        return L
    
# ---------------------- A utility function to generate points ----------------- #

def gen_data_points_close_to_true(p0, like_func, priors, ntemps, nwalkers, ndims, source_type='bh', stdtol=5.0, factor=1e-5, max_iter=1000, iter_check=0):
    """ A silly function to initialize the walkers at points close
        to the true values. It uses the spread of the likelihodo as 
        a criterion.

    Raises:
        ValueError: When it fails to generate the points, usually something 
        is wrong with the prior densities as defined by the user. Use with care.
    """
    # generate starting points
    cov = np.ones(ndims) * 1e-3
    start_like = np.zeros((nwalkers * ntemps))
 
    while np.std(start_like) < stdtol:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndims))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
            if source_type=='bh':
                tmp[fix] = (p0[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndims)))[fix]
                tmp[:, 10] = tmp[:, 10] % (2 * np.pi) # phi0
                tmp[:, 8] = tmp[:, 8] % (2 * np.pi) # lamda
                tmp[:, 9] = tmp[:, 9] % (1 * np.pi) # psi
            elif source_type=='ucb':
                tmp[fix] = (p0[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndims)))[fix]
                tmp[:, 3] = tmp[:, 3] % (2 * np.pi) # phi0
                tmp[:, 6] = tmp[:, 6] % (2 * np.pi) # lamda
                tmp[:, 5] = tmp[:, 5] % (1 * np.pi) # psi
            else:
                raise TypeError("Error: Please choose between 'bh' 'ucb', and 'stochastic'.")
                
            logp = priors.logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        start_like = like_func(tmp)

        iter_check += 1
        factor *= 1.5

        print('\t ({}) std[LLH] = {}'.format(iter_check, np.std(start_like)))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")
        
    return tmp.reshape(( ntemps, nwalkers, ndims))


# ---------------------- A utility function to get the clean chains ----------------- #

def get_clean_chain(coords, ndim, temp=0):
    """ A silly function to get the clean parameter chains out. 
        In the future it will not be needed.
    """
    naninds    = np.logical_not(np.isnan(coords[:, temp, :, :, 0].flatten()))
    print(np.sum(naninds))
    samples_in = np.zeros((coords[:, temp, :, :, 0].flatten()[naninds].shape[0], ndim))  # init the chains to plot
    # get the samples to plot
    for d in range(ndim):
        givenparam = coords[:, temp, :, :, d].flatten()
        samples_in[:, d] = givenparam[
            np.logical_not(np.isnan(givenparam))
        ]  # Discard the NaNs, each time they change the shape of the samples_in
    return samples_in

# ---------------------- A utility function to compute XYZ -> AET ----------------- #

def aet(X,Y,Z, factor=10**19):
    """ Simple transform from XYZ to AET function.
    """
    X, Y, Z = X*factor, Y*factor, Z*factor
    A, E, T = (Z - X)/np.sqrt(2.0), (X - 2.0*Y + Z)/np.sqrt(6.0), (X + Y + Z)/np.sqrt(3.0)
    return A/factor, E/factor, T/factor

# END