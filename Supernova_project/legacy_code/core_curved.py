import jax
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro import sample
import multiprocessing
import os

# Constants
c = 299792.458  # speed of light in km/s

def H(z, H0, Om, Ok):
    """Calculate Hubble parameter with safeguards"""
    # Ensure inputs are positive where needed
    H0 = jnp.maximum(H0, 1.0)  # Ensure H0 is positive
    Om = jnp.clip(Om, 0.0001, 0.9999)  # Ensure Om is positive but < 1
    
    # Calculate Ol (dark energy density) ensuring it's not negative
    Ol = jnp.maximum(1.0 - Om - Ok, 0.0)
    
    # Calculate E(z)
    E = jnp.sqrt(Om * (1 + z)**3 + Ok * (1 + z)**2 + Ol)
    
    return H0 * E

def luminosity_distance(z, Om, H0, Ok):
    """Keep original integration but add minimal safeguards"""
    N = 100  # Keep your original number of points
    z_array = jnp.linspace(0, z, N)
    dz = z_array[1] - z_array[0]
    
    integrand_values = vmap(lambda z_: c / H(z_, H0, Om, Ok))(z_array)
    chi = jnp.sum(integrand_values) * dz
    
    # Original curvature handling with minimal safeguards
    if_flat = (1 + z) * chi
    if_pos = (1 + z) * (1/jnp.sqrt(jnp.abs(Ok))) * jnp.sinh(jnp.sqrt(jnp.abs(Ok))*chi)
    if_neg = (1 + z) * (1/jnp.sqrt(jnp.abs(Ok))) * jnp.sin(jnp.sqrt(jnp.abs(Ok))*chi)
    
    result = jnp.where(jnp.abs(Ok) < 1e-10, if_flat,
                jnp.where(Ok > 0, if_pos, if_neg))
    
    return jnp.clip(result, 1e-10, 1e15)  # Just add final clipping

def distance_modulus(z, Om, H0, Ok):
    """Calculate distance modulus"""
    dL = luminosity_distance(z, Om, H0, Ok)

    # Clip the luminosity distance to prevent infinities
    dl_clipped = jnp.clip(dL, 1e-10, 1e15)  # Upper limit prevents inf in log
    # return 55 if dL is infinite
    return 5 * jnp.log10(jnp.maximum(dl_clipped,1e-10)) + 25

# Vectorize the distance modulus calculation
distance_modulus_vec = vmap(distance_modulus, in_axes=(0, None, None, None))

def model(z, mu_obs, mu_err):
    # Make prior relativly narrow
    H0 = sample("H0", dist.Uniform(60, 80))     # Uniform prior on H0
    Om = sample("Om", dist.Uniform(0.1, 0.9))   # Uniform prior on Omega_m
    Ok = sample("Ok", dist.Uniform(-0.1, 0.1)) # Uniform prior on Omega_k
    
    # Calculate expected distance modulus
    mu_exp = distance_modulus_vec(z, Om, H0, Ok)
    
    # Likelihood (assuming independent measurements)
    sample("mu_obs", dist.Normal(mu_exp, mu_err), obs=mu_obs)  