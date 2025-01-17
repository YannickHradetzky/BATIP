import os

def set_jax_config():
    # Force CPU and single-thread for stability
    os.environ['JAX_PLATFORM_NAME'] = 'METAL'
    os.environ['JAX_ENABLE_X64'] = 'True'  # Enable 64-bit precision
    
    # Use single thread for better stability
    desired_cores = 10
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={desired_cores}"
    os.environ["OPENBLAS_NUM_THREADS"] = str(desired_cores)
    os.environ["MKL_NUM_THREADS"] = str(desired_cores)
    
    import jax
    jax.config.update("jax_enable_x64", True)  # Double ensure 64-bit precision
    jax.config.update("jax_platform_name", "cpu")
    print(f"Using CPU backend for JAX with {desired_cores} core")
    print("Available devices:", jax.devices())
    
# Make sure this is called before any other imports
set_jax_config()

import jax
import jax.numpy as jnp
import numpy as np

from core_curved import model
import numpyro
import numpyro.distributions as dist
from numpyro import sample
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS

def load_data():
    df = pd.read_csv('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Pantheon+SH0ES.dat', sep='\s+', header=0)
    return df



if __name__ == "__main__":
    set_jax_config()
    df = load_data()
    z = df['zHD'].values
    mu = df['MU_SH0ES'].values
    mu_err = df['MU_SH0ES_ERR_DIAG'].values

    rng_key = jax.random.PRNGKey(42)
    
    # Modify the NUTS kernel configuration
    kernel = NUTS(
        model,
        target_accept_prob=0.8,
        max_tree_depth=7,  # Increased from 1
        step_size=0.1      # Decreased from 0.1
    )
    
    mcmc = MCMC(
        kernel,
        num_warmup=1000,    # Increased warmup steps
        num_samples=2000,
        progress_bar=True
    )
    
    # Run with multiple chains for better diagnostics
    n_chains = 4           # Increased from 1
    rng_key = jax.random.split(rng_key, n_chains)
    mcmc.run(
        rng_key, 
        z, mu, mu_err,
        extra_fields=('potential_energy', 'diverging')
    )
    mcmc.print_summary()
    
    samples = mcmc.get_samples()

    # from this distribution calculate Omega_lambda
    Omega_lambda = 1 - samples['Om'] - samples['Ok'] 
    
    # Create a figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    
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

    # Plot Omega_k distribution
    ax3.hist(samples['Ok'], bins=50, color='salmon', edgecolor='black', density=True)
    ax3.set_title('Ωₖ Distribution')
    ax3.set_xlabel('Ωₖ')
    ax3.set_ylabel('Frequency')
    
    # Plot Omega_lambda distribution
    ax4.hist(Omega_lambda, bins=50, color='purple', edgecolor='black', density=True)
    ax4.set_title('Ωₗ Distribution')
    ax4.set_xlabel('Ωₗ')
    ax4.set_ylabel('Frequency')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()