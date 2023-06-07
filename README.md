# Investigating heavier tailed likelihoods for robust inference in Gravitational Wave Astronomy

This is a repository where we investigate different statistical strategies for robust inference agains data outliers, focusing on applications on Gravitational Wave Astronomy. In particular, we apply the Hyperbolic likelihood filter to different noise-knowledge scenarios. The associated paper can be found at [2305.04709](https://arxiv.org/abs/2305.04709).

N Karnesis, A Sasli, N Stergioulas 2022

--------
##### Installation 

First we need to create a `conda` environment:
```
conda create -n gpu_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl lapack=3.6.1 numpy scipy Cython cupy jupyter ipython matplotlib python=3.9
```
If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`. We then activate the environment:
```
conda activate gpu_env
```
Then, we need to have `cupy` installed. This is already taken cared of from our conda environment, but sometimes we might need a specific version of `cupy`. For example, for our local machine, the snipped below has worked 
```
pip install cupy-cuda11x
```
after having activated our conda environment of course. Now, We also need to export the `cuda` paths in our `.basrc` or `.bash_profile` files, something like
```
export CUDAHOME=/usr/local/cuda-11.6
alias nvcc=/usr/local/cuda-11.6/bin/nvcc
```
For example, for our `lpf.astro.auth.gr` machine I have set
```
export CUDAHOME=/usr/local/cuda-12.0
export PATH=${PATH}:/usr/local/cuda-12.0/bin
alias nvcc=/usr/local/cuda-12.0/bin/nvcc
```
There are a few more packages that are useful to install via `pip`, these are the 
```
h5py, tqdm, corner, chainconsumer, torch
```
Finally, we will need the [`Eryn`](https://github.com/mikekatz04/Eryn) sampler. Just `git clone` the `dev` branch of the repository and then install the package via navigating into the directory and running
```
python setup.py install
```
Especially for the `dev` branch, you might need to edit the `setup.py` file, and add `"-std=c++17",` inside each `extra_compile_args` dictionary. 

In the end, it is also good to have the [`LDC`](https://gitlab.in2p3.fr/LISA/LDC) software installed, because we can access different types of waveforms, or LISA noise curves. This can be done either by flollowing the instructions on the website, or simply via
```
pip install lisa-data-challenge
```
--------
##### Ultra Compact Galactic Binaries

In principle, we can already use the waveforms from the LDC software. But why not take advantage of our GPUs, and use the [`GBGPU`](https://github.com/mikekatz04/GBGPU) package instead? Same as before, we `git clone` the `main` branch of the repository and we simply navigate into its directory and run 
```
python setup.py install
```
All the prerequisite packages for `GBGPU` have been preinstalled in our `gpu_env` conda environmet. If we want to install the `dev` branch, we will need also the [`mathdx`](https://developer.nvidia.com/mathdx) package. We can either download the `cufftx` directory and put it inside the `GBGPU` directory, and/or  add the correct directory into the `include_dirs` variable in the `setup.py` file (after having installed the package for all users). It should look like something like this:
```
"/usr/local/mathdx/nvidia-mathdx-22.11.0-Linux/nvidia/mathdx/22.11/example/cufftdx"
"/usr/local/mathdx/nvidia-mathdx-22.11.0-Linux/nvidia/mathdx/22.11/include/cufftdx/include"
``` 
All this is doen becasue the code is under development. In the near future, installation will be much easier. 

--------
### Usage 

We have created a main scripts, the `hyperbolic_likelihood_inv_ucbs.py` that is used for 
* Setting up the source parameters [Ultra Compac Galactic Binaries]
* Generating the data using a specific random seed. 
* Setting the sampler specifics [number of walkers, temperatures and total samples per walker]
* Make some basic plots for the results.

We run the scripts as
```
python likelihood_inv_ucbs_script.py 1 2 3 4
```
where the input numbers correspond to the likelihood type to be run. The available types are the
1. Gaussian likelihood, known noise, correct levels
2. Gaussian likelihood, known noise, false levels [levels can be tuned]
3. Gaussian likelihood, fitting the noise
4. Hyperbolic likelihood, known noise, false levels  [levels can be tuned]

#### Outputs of `hyperbolic_likelihood_inv_ucbs.py` script

For instance by running `python hyperbolic_likelihood_inv_ucbs.py 1 2 3 4`, the files `[TAG]_chains_llhType1.png`, `[TAG]_chains_llhType2.png`, `[TAG]_chains_llhType3.png`, and `[TAG]_chains_llhType4.png` are created.
- `[TAG]_cornerplot_llhType[likelihood_case].png`: a corner plot figure for a specific lekilehood case
- `[TAG]_data.png`: a figure showing the "Generating data", the "signal", the correct "Noise PSD" and the "Wrong noise PSD"
- `[TAG]smbh_low_snr_signal_only.png`: a figure showing the signal (in frequency domain) in A, E, T channels

2. **`chains` directory**
- `[TAG]_mhmcmc_chains_llhType[likelihood_case].h5`: the chains for a specific likelihood case in h5 file format

![Alt text](py/figs/demo.png?raw=true)
