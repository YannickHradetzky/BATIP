import os
import jax
import jax.numpy as jnp
import numpyro

def set_jax_config():
    jax.config.update("jax_enable_x64", True)  # Metal GPU requires float32
    jax.config.update("jax_platform_name", "cpu")  # Enable Metal backend
    # Set number of CPU devices for parallel chains
    numpyro.set_host_device_count(4)

    print("JAX version:", jax.__version__)
    print("Available devices:", jax.devices())
    print("Default device:", jax.default_backend())
    print("Number of devices for parallel chains:", jax.local_device_count())
    
# Make sure this is called before any other imports
set_jax_config()

import numpy as np
from core_flat import model, plot_samples
import numpyro.distributions as dist
from numpyro import sample
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES.dat', sep='\s+', header=0)
    cov_matrix = pd.read_csv('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES_STAT+SYS.cov')

    z = df['zCMB'].values
    mu_obs = df['m_b_corr'].values
    cov_matrix = cov_matrix.values.reshape(len(df), len(df))
    
    # Force symmetry by averaging with its transpose
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    
    # Debug information
    print("Covariance matrix shape:", cov_matrix.shape)
    print("Any NaN values:", np.any(np.isnan(cov_matrix)))
    print("Any infinite values:", np.any(np.isinf(cov_matrix)))
    print("Is symmetric:", np.allclose(cov_matrix, cov_matrix.T))
    
    # Check eigenvalues
    eigenvals = np.linalg.eigvals(cov_matrix)
    print("Minimum eigenvalue:", np.min(eigenvals))
    print("Maximum eigenvalue:", np.max(eigenvals))
    
    # Add small diagonal term for numerical stability
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-6
    
    # Convert to device array
    cov_matrix = jnp.array(cov_matrix)
    
    return z, mu_obs, cov_matrix

if __name__ == "__main__":
    # Load and prepare data
    z, mu_obs, cov_matrix = load_data()
    
    # Initialize random key
    rng_key = jax.random.PRNGKey(0)
    
    # NUTS kernel with simpler settings
    kernel = numpyro.infer.NUTS(
        model,
        target_accept_prob=0.8,
        init_strategy=numpyro.infer.initialization.init_to_sample(),  # Changed to sample-based init
        step_size=0.1,  # Larger step size
        adapt_mass_matrix=True,
        max_tree_depth=8
    )

    # MCMC settings
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1000,
        num_chains=1,  # Start with single chain for debugging
        progress_bar=True
    )
    
    # Run MCMC
    mcmc.run(rng_key, z, mu_obs, cov_matrix)
    
    # Print summary
    mcmc.print_summary()
    
    # Print mean values
    samples = mcmc.get_samples()
    print("\nMean values:")
    print(f"H0 = {jnp.mean(samples['H0']):.2f} km/s/Mpc")
    print(f"Om = {jnp.mean(samples['Om']):.3f}")
