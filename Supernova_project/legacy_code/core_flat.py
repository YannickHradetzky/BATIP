import jax
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro import sample
import multiprocessing
import os
import matplotlib.pyplot as plt

# Constants
c = 299792.458  # speed of light in km/s

def set_default_dtype():
    """Set default floating-point precision to float64"""
    jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(precision=8)

def H(z, H0, Om):
    """Hubble parameter at redshift z"""
    OL = 1 - Om
    return H0 * jnp.sqrt(Om * (1 + z)**3 + OL)

def luminosity_distance(z, Om, H0):
    """Calculate luminosity distance in Mpc"""
    # Use fewer points but with better spacing
    N = 100
    z_array = jnp.logspace(-3, jnp.log10(1 + z), N) - 1  # Better sampling near z=0
    dz = jnp.diff(z_array, append=z_array[-1])
    
    # Calculate integrand
    integrand_values = c / H(z_array, H0, Om)  # c is in km/s
    chi = jnp.sum(integrand_values * dz)
    
    return (1 + z) * chi

def distance_modulus(z, Om, H0):
    """Calculate distance modulus"""
    dL = luminosity_distance(z, Om, H0)
    return 5 * jnp.log10(jnp.maximum(dL, 1e-10)) + 25

# Vectorize the distance modulus calculation
distance_modulus_vec = vmap(distance_modulus, in_axes=(0, None, None))

def check_physical_constraints(z, Om, H0):
    """Verify physical constraints of the model"""
    checks = {
        "H0 positive": jnp.array(H0 > 0, dtype=jnp.float64),
        "Om in [0,1]": jnp.array((0 <= Om) & (Om <= 1), dtype=jnp.float64),
        "z non-negative": jnp.array(z >= 0, dtype=jnp.float64),
        "H(z) real": jnp.array(H(z, H0, Om) > 0, dtype=jnp.float64),
        "dL positive": jnp.array(luminosity_distance(z, Om, H0) > 0, dtype=jnp.float64)
    }
    
    return checks

def model(z, mu_obs, cov_matrix):
    """Model with physically motivated priors"""
    # Priors based on current cosmological constraints
    H0 = sample("H0", dist.Uniform(65.0, 75.0))      # km/s/Mpc
    Om = sample("Om", dist.Uniform(0.2, 0.4))        # Current consensus range
    
    # Calculate expected distance modulus
    mu_exp = distance_modulus_vec(z, Om, H0)
    
    # Likelihood
    sample("obs", dist.MultivariateNormal(mu_exp, cov_matrix), obs=mu_obs)
    
    # If you want to track the constraints, you can sample them
    # This will show up in your diagnostics
    for name, check in check_physical_constraints(jnp.max(z), Om, H0).items():
        sample(f"check_{name}", dist.Delta(check))

def plot_samples(samples):
        # from this distribution calculate Omega_lambda
    Omega_lambda = 1 - samples['Om'] 
    
    # Create a figure with four subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot H0 distribution
    ax1.hist(samples['H0'], bins=50, color='skyblue', edgecolor='black', density=True)
    ax1.set_title('H₀ Distribution')
    ax1.set_xlabel('H₀ [km/s/Mpc]')
    ax1.set_ylabel('Frequency')
    
    # Plot Omega_m distribution
    ax2.hist(samples['Om'], bins=50, color='lightgreen', edgecolor='black', density=True)
    ax2.set_title('Ωₘ Distribution')
    ax2.set_xlabel('Ωₘ')
    ax2.set_ylabel('Frequency')
    
    # Plot Omega_lambda distribution
    ax3.hist(Omega_lambda, bins=50, color='salmon', edgecolor='black', density=True)
    ax3.set_title('Ωₗ Distribution')
    ax3.set_xlabel('Ωₗ')
    ax3.set_ylabel('Frequency')

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('Plots/flat_universe_samples.png')
    plt.close()

def universe_age(Om, H0):
    """Calculate age of the universe in Gyr"""
    # Integration from z=0 to infinity (or large z)
    z_max = 1000
    N = 1000
    z_array = jnp.logspace(-2, jnp.log10(z_max), N)
    dz = jnp.diff(z_array)
    z_mid = (z_array[1:] + z_array[:-1]) / 2
    
    integrand = 1 / (H(z_mid, H0, Om) * (1 + z_mid))
    age = jnp.sum(integrand * dz) * 977.8  # Convert to Gyr
    
    return age
