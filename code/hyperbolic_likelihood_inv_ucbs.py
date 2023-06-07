# ---------------------------------------------------------------------- #
#         Investigating the Hyperbolic filter as likelihood function 
# ---------------------------------------------------------------------- #

import numpy as np
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt

# add singularity-image path for lisabeta and eryn
import sys, os, copy
import time

# useful modules for sampling etc
from chainconsumer import ChainConsumer
from eryn import ensemble
from eryn.ensemble import EnsembleSampler
from eryn.backends import HDFBackend
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.utils import TransformContainer
from lisatools.sensitivity import get_sensitivity

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice
    gpu_available = True
    print("\n > Found CUPY, I'll try to use it!\n")
except ModuleNotFoundError:
    import numpy as xp
    gpu_available = False
    
# Load the local tools 
from statutils import *

# get input arguments
argv = sys.argv

scriptname = argv[0].split('.')[0]

print(' > Running {}.py \n'.format(scriptname))

if len(argv) < 2:
    sys.exit(" ### Error: I need 1 inputs: Just give me the type of likelihood to run! Choose between N [1, 5]")
else:
    input_like_type_inv = []
    for i in range(1, len(argv)):
        input_like_type_inv.append(int(argv[i]))

#------------------------------------- Run Settings Below ----------------------------- #

like_type_inv = input_like_type_inv
                  # Or choose by hand between the selections below:
                  # 1 : Gaussian likelihood, known noise, correct levels
                  # 2 : Gaussian likelihood, known noise, false levels
                  # 3 : Gaussian likelihood, fitting the noise
                  # 4 : Hyperbolic likelihood, known noise, false levels

DOPLOT   = True
burn     = 20000
Nsmpls   = 60000
nwalkers = 50
ntemps   = 30
TAG      = 'a_tag_here' # An overal tag for the run
rseed    = 1234 # The random seed

# WF parameters
log10A = -21.37729133
f0 = 2.61301 # In [mHz]
log10fdot = -16.53675992
phi0 = 3.12184095
cosi = 0.05407993
psi = 0.80372815
lamda = 1.76872
sinb = 0.1012303

# Noise parameters: Nominal values for the SciRD noise
Sa = np.log10((3.e-15)**2)  # m^2/sec^4/Hz
Si = np.log10((15.e-12)**2) # m^2/Hz

# Wrong values for the SciRD noise
Sa_wrong = np.log10((3.8e-15)**2)  # m^2/sec^4/Hz
Si_wrong = np.log10((32.e-12)**2) # m^2/Hz

t_obs = 1 # 1/12 # YRS
dt = 15 # The cadence, in seconds. Used to get the maximum frequency, also an input to the waveform

#------------------------------------- Run Settings Above ----------------------------- #

print(" > Running on environment '{}'\n".format(os.environ['CONDA_DEFAULT_ENV']))
 
# make the plots look a bit nicer with some defaults
rcparams = {}
rcparams['axes.linewidth'] = 0.5
rcparams['font.family'] = 'serif'
rcparams['font.size'] = 22
rcparams['legend.fontsize'] = 16
rcparams['mathtext.fontset'] = "stix"
#rcparams['agg.path.chunksize'] = 1000000
mpl.rcParams.update(rcparams) # update plot parameters
mpl.rcParams['agg.path.chunksize'] = 10000

# Define some nice colours
clrs = ["#455A64", "#B32222", "#D1D10D"]

# make dir for figs
if not os.path.exists("figs"):
    os.makedirs("figs")
if not os.path.exists("chains"):
    os.makedirs("chains")

# add time to estimate the elapsed time
start_time = time.time()

# ----------------------------------------- Data Gen: ------------------------------------- #
#

# set random seed. Important in order to sample the same data!
xp.random.seed(rseed)

print(" > Generating some data with random seed = {}\n".format(rseed))

df = 1/(t_obs*YRSID_SI) # Get the frequency resolution

wf_parameter_names = [r"$\log_{10}\mathcal{A}$", r"$f_\mathrm{gw}~[\mathrm{mHz}]$", \
                      r"$\log_{10}\dot{f}_0$", r"$\phi_0$", r"$\cos\iota$", \
                      r"$\psi$", r"$\lambda$", r"$\sin\beta$"]

p0 = np.array([[log10A, f0, log10fdot, phi0, cosi, psi, lamda, sinb]])

# Get the default test-parameter values
print("\n > The default parameters are:\n")
print(" logA: {} \n   f0: {} \n fdot: {} \nfddot: {} \n phi0: {} \n \
  ci: {} \n  psi: {} \n  lam: {} \n   si: {}\n".format(*np.insert(p0, 3, 0.0, axis=1)[0,:]))

# Get the waveform settings
f0_lims = np.array([f0-1e-3, f0+1e-3]) * 1e-3 # <

buffer = 200 # N-points as a buffer
fmin = f0_lims[0] - buffer * df
fmax = f0_lims[1] + buffer * df
start_freq_ind = int(fmin / df)
end_freq_ind = int(fmax / df)

fvec = np.arange(start_freq_ind, end_freq_ind + 1) * df # Our frequency vector

# Get the PSD of the noise using hte model above
lisa_noise_model = noise_model(fvec)
Sn = lisa_noise_model([Sa, Si])

# Get the PSD of the noise using the model above
Sn_wrong = lisa_noise_model([Sa_wrong, Si_wrong])

# Calculate noise ratio
n_ratio =  np.squeeze(noise_model(f0*1e-3).eval([Sa, Si]) / noise_model(f0*1e-3).eval([Sa_wrong, Si_wrong]))

print(" > Will assume wrong noise levels by a factor of {}\n".format(n_ratio))

# ------------------------------ Generate some UCBs signal --------------------------- #      
#

# Evaluate the model - If GPU is found we'll get cuda arrays
template = ucb(f_lims=f0_lims, f_inds=[start_freq_ind, end_freq_ind + 1], \
                gpu=gpu_available, tobs=t_obs*YRSID_SI, dt=dt)
As, Es, _ = template.eval(p0)

# Ensure we have cupy arrays for the case we use GPUs
As = xp.array(As)
Es = xp.array(Es)

print(" > Generating some random noise for the real data\n")

# Generate some random noise for the real data
n_real     = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))
n_imag     = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))

# Ensure we wrap cuda arrays in case a GPU is found
data_noise = xp.array(n_real) + xp.array(n_imag)
data_A     = xp.array(n_real) + xp.array(n_imag) + As

n_real     = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))
n_imag     = xp.random.normal(scale=xp.sqrt(lisa_noise_model([Sa, Si]))/(2.0*xp.sqrt(df)))

# Ensure we wrap cuda arrays in case a GPU is found
data_noise = xp.array(n_real) + xp.array(n_imag)
data_E     = xp.array(n_real) + xp.array(n_imag) + Es

# -------------------------------- Make a test plot ------------------------ #
# 

if DOPLOT:
    plt.figure()
    plt.loglog(fvec, np.abs(As.squeeze().get()), label='A')
    plt.loglog(fvec, np.abs(Es.squeeze().get()), label='E')
    plt.legend()
    # plt.xlim([fmin, fmax])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$\tilde{h}(f)$ (Hz$^{-1/2}$)")
    plt.savefig('figs/{}_ucbs_signal_only.png'.format(TAG), dpi=600, bbox_inches='tight')

    plt.figure(figsize=(16,6))
    plt.loglog(fvec, 2*df*np.absolute(data_A.squeeze().get())**2 , label='Generated Data', alpha=0.3, color='grey')
    plt.loglog(fvec, 2*df*np.absolute(As.squeeze().get())**2 , label='signal', alpha=0.5, color='limegreen')
    plt.loglog(fvec, Sn.squeeze().get(), label='Noise PSD',color='k', linestyle='-.')
    plt.loglog(fvec, Sn_wrong.squeeze().get(), label='Wrong noise PSD',color='r', linestyle=':')
    plt.ylabel('[1/Hz]')
    plt.xlabel('Frequency [Hz]')
    plt.xlim([fvec[1], fvec[-1]])
    plt.ylim(1e-43, 1e-38)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('figs/{}_ucbs_data.png'.format(TAG), dpi=600, bbox_inches='tight')

# -------------------------------- Compute the SNR ------------------------ #

SNR2 = lambda d, s: (4.0*df) * xp.sum(xp.real((d*xp.conjugate(s))/Sn))
print(' # SNR: ', xp.sqrt(SNR2(data_A, As) + SNR2(data_E, Es)))

# ----------------------------- Define the distribution class ---------------------------- #
#

print("\n > Chose likelihood investigation type '{}'\n".format(like_type_inv))


# ----------------------------- Define the distribution class ---------------------------- #
#
#   1 : Gaussian likelihood, known noise, correct levels
#   2 : Gaussian likelihood, known noise, false levels
#   3 : Gaussian likelihood, fitting the noise
#   4 : Hyperbolic likelihood, known noise, false levels
#   5 : Hyperbolic likelihood, known noise, correct levels

pgen = copy.deepcopy(p0) # Copy the wf initial values

for like_type in like_type_inv:
    
    print("\n > Running likelihood investigation type '{}'\n".format(like_type))

    # Setup priors for WF parameters -> names :[log10A, f0, log10fdot, phi0, cosi, psi, lamda, sinb]
    cmin = np.array([-30, .1*f0,  -20., 0.,      -1, 0.0,    0.,     -1])
    cmax = np.array([-19, 2.0*f0, -10., 2*np.pi, 1., np.pi,  2*np.pi, 1])

    llh_stdtol=5.0   # These are default settings for generating initial points for each walker
    llh_factor=1e-5
    
    if like_type == 1: # 1 : Gaussian likelihood, known noise, correct levels
        
        # Choose the log-likelihood type
        distr = loglike([data_A, data_E], template, fvec, df, Sn, gpu=gpu_available)

        # Choose the log-likelihood type
        logL = distr.gaussian
        
        # Mark the true parameters
        truth_params = list(p0[0,:])
        
        # Set the parameter names
        parameter_names = wf_parameter_names
        
    elif like_type == 2: # 2 : Gaussian likelihood, known noise, false levels
        
        # Choose the log-likelihood type
        distr = loglike([data_A, data_E], template, fvec, df, Sn_wrong, gpu=gpu_available)

        # Choose the log-likelihood type
        logL = distr.gaussian
        
        # Mark the true parameters
        truth_params = list(p0[0,:])
        
        # Set the parameter names
        parameter_names = wf_parameter_names

    elif like_type == 3: # 3 : Gaussian likelihood, fitting the noise

        # Add the extra parameter here
        p0 = np.append( pgen[0], n_ratio.get() )[None,:]
        parameter_names = wf_parameter_names + [r'$n_\mathrm{level}$'] 

        # Choose the log-likelihood type
        distr = loglike([data_A, data_E], template, fvec, df, Sn_wrong, gpu=gpu_available)

        # Choose the log-likelihood type
        logL = distr.gaussian_noise_fit_level
        
        # Mark the true parameters
        truth_params = list(pgen[0,:]) +  [n_ratio]
        
        # Add prior ranges for extra parameters
        cmin = np.append(cmin, np.array([0.001]) )
        cmax = np.append(cmax, np.array([10]) )

    elif like_type == 4: # 4: Hyperbolic likelihood, known noise
        
        # Choose the log-likelihood type (between 4 & 5)
        distr = loglike([data_A, data_E], template, fvec, df, Sn, gpu=gpu_available)

        # Extract the log-likelihood function
        logL = distr.hyperbolic
        
        alpha_delta_ini_params = [1,1] # Initialize log_10(alpha) and log_10(delta) to unity
        
        # Add the extra parameter here
        p0 = np.append( pgen[0], np.array(alpha_delta_ini_params) )[None,:]
        parameter_names = wf_parameter_names + [r'$\log_{10}\alpha$',r'$\log_{10}\delta$'] 
        
        # Mark the true parameters
        truth_params = list(pgen[0,:]) +  [np.nan, np.nan]
        
        # Add prior ranges for extra parameters
        cmin = np.append(cmin, np.array([-310., -310.]) )
        cmax = np.append(cmax, np.array([100., 100.]) )
        
        # Change those factors for type 4 & 5
        llh_stdtol=1e-2 
        llh_factor=1e-15

        
    # Test the likelihood for multiple binaries computation
    like_func_to_use = logL
    # Test the GPU likelihood by ccomputing multiple parameter sets at once
    num_bins = 10
    test_params_in = np.tile(p0, (num_bins, 1))
    _ = like_func_to_use(test_params_in)

    # Eval llh at true values
    print(' > Log-likelihood value for choice {}: {}'.format(like_type, like_func_to_use(p0).squeeze()))
    ndims = p0.shape[1] # <--- number of dimensions

    # ------------------------------ Priors - Define priors ------------------------------ #
    #  

    priors_in = {i: uniform_dist(cmin[i], cmax[i]) for i in range(ndims)}
    priors = ProbDistContainer(priors_in)

    # Set the periodic parameters [log10A, f0, log10fdot, phi0, cosi, psi, lamda, sinb]
    periodic = {"model_0": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}

    # starting positions 
    starting_vals = gen_data_points_close_to_true(p0.squeeze(), like_func_to_use, priors, \
                                                    ntemps, nwalkers, ndims, source_type='ucb', \
                                                        stdtol=llh_stdtol, factor=llh_factor)

    print(" > Found some starting parameters for all walkers.")

    coords = np.zeros((ntemps, nwalkers, 1, ndims)) # Define the starting coordinates
    coords[:, :, 0, :] = starting_vals.copy()

    # Evaluate at the initial values
    log_l = like_func_to_use(coords[:, :, 0, :].reshape(ntemps*nwalkers, ndims))

    # Define the initial state, plus the H5 file to save the data to.
    state = State(coords, log_like=log_l.reshape(ntemps, nwalkers)) # blobs=blobs
    betas = np.linspace(1.0, 0.0, ntemps)
    backend = HDFBackend("chains/{}_mhmcmc_chains_llhType{}.h5".format(TAG,like_type))

    # Define the Ensemble sampler
    ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            like_func_to_use,
            priors,
            args=(),
            vectorize=True,
            tempering_kwargs=dict(betas=betas, stop_adaptation=burn),
            plot_iterations=-1,
            periodic=periodic,
            update_iterations=1,
            backend=backend,
        )

    print(" > Sampling starting ...")

    ensemble.run_mcmc(state, int(Nsmpls), burn=burn, progress=True, thin_by=1)

    print(" > Sampling finished.")

    # Get the MCMC chains out
    mcmc_samples = get_clean_chain(ensemble.get_chain()['model_0'], ndims)

    # ----------------------------------- Do the MCMC-chains plot --------------------------- #
    #

    pnames_for_plot = wf_parameter_names

    fig, axs = plt.subplots(len(pnames_for_plot), 1, figsize=(12,16), sharex=True)
    for i in range(len(pnames_for_plot)):
        axs[i].plot(mcmc_samples[:,i], '#455A64', alpha=.5)
        axs[i].set_ylabel(pnames_for_plot[i])
        axs[i].axhline(y=p0[0][i], color='crimson', linestyle='-')
    axs[-1].set_xlabel('Samples')
    axs[-1].set_xlim(0, mcmc_samples.shape[0])
    plt.savefig('figs/{}_chains_llhType{}.png'.format(TAG,like_type), dpi=600)

    # ----------------------------------- Do the corner-plot -------------------------------- #
    #

    ch = ChainConsumer()
    ch.add_chain(mcmc_samples, parameters=parameter_names, name=TAG)
    ch.configure(colors=clrs, shade=True, shade_alpha=0.2, bar_shade=True)
    ch.plotter.plot(figsize=(12,12), truth=truth_params, filename="figs/{}_cornerplot_llhType{}.png".format(TAG,like_type))

# ------------------------------ END -----------------------------
#

elapsed = (time.time() - start_time)
print("\n > END. Elapsed time: {}\n".format(elapsed))

# END
