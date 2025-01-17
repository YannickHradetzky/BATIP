import pandas as pd
import numpyro
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import sample
import matplotlib.pyplot as plt
# Set this at the beginning of your script
jax.config.update('jax_platform_name', 'cpu')  # Force CPU
numpyro.set_host_device_count(4)  # Match the number of chains we want to run

# load the data 
df = pd.read_csv('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES.dat', sep='\s+', header=0)
# load the covariance matrix
cov_data_stat = np.loadtxt('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES_STATONLY.cov')
cov_data_sys_stat = np.loadtxt('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES_STAT+SYS.cov')

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
    plt.savefig('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/flat_posteriors.png')
    plt.close()

def create_cov_mat(cov_data, title):
    # First value is the matrix size
    n = int(cov_data[0])  # Should be 1701
    cov_matrix = cov_data[1:].reshape(n, n)

    # Verify it's symmetric
    is_symmetric = np.allclose(cov_matrix, cov_matrix.T)
    if not is_symmetric:
        print("Matrix is not symmetric")
        # make it symmetric
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        if not np.allclose(cov_matrix, cov_matrix.T):
            print("Matrix is still not symmetric")
            return None
    else:
        print("Matrix is symmetric")

    # Check positive definiteness
    eigenvalues = np.linalg.eigvals(cov_matrix)
    is_positive_definite = np.all(eigenvalues > 0)
    print(f"Matrix is positive definite: {is_positive_definite}")
    # plot the covariance matrix
    plt.imshow(cov_matrix, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/{title}_cov_matrix.png')
    plt.close()
    return cov_matrix


# get the data
z = df['zCMB'].values
mu_obs = df['MU_SH0ES'].values
mu_err = df['MU_SH0ES_ERR_DIAG'].values

# create the covariance matrix
sys_stat_cov_matrix = create_cov_mat(cov_data_sys_stat, 'sys_and_stat')
stat_cov_matrix = create_cov_mat(cov_data_stat, 'stat_only')

# After loading data, but before the model definition
print("Original data points:", len(z))

# Filter out very low redshift points
mask = z > 0.001
z = z[mask]
mu_obs = mu_obs[mask]
mu_err = mu_err[mask]

print("Filtered data points:", len(z))


# -------------------------------------------------------------------------------------------------
# Physics
# -------------------------------------------------------------------------------------------------
c = 299792.458 # km/s

def hubble_z(H0, Om, Ok, z):
    """Hubble parameter at redshift z in km/s/Mpc"""
    matter_term = Om * (1 + z)**3
    lambda_term = (1 - Om)  # Since Ok = 0
    return H0 * jnp.sqrt(matter_term + lambda_term)

def chi(h0, Om, Ok, z):
    """Comoving distance in Mpc"""
    N = 1000
    z_max = jnp.max(z)
    z_array = jnp.linspace(0.001, z_max, N)
    dz = jnp.diff(z_array, append=z_array[-1])
    
    # Calculate integrand (c is already in km/s)
    integrand_values = c / hubble_z(h0, Om, Ok, z_array)
    
    # Integrate
    return jax.vmap(lambda z_point: jnp.sum(integrand_values * dz * (z_array <= z_point)))(z)

def luminosity_distance(H0, Om, Ok, z):
    """Luminosity distance in Mpc"""
    chi_val = chi(H0, Om, Ok, z)
    return (1 + z) * chi_val  # Flat universe case

def distance_modulus(H0, Om, Ok, z):
    """Distance modulus in magnitudes"""
    d_L = luminosity_distance(H0, Om, Ok, z)
    return 5 * jnp.log10(d_L) + 25

def test_cosmological_calculations(H0, Om, Ok, title):
    """Test our cosmological calculations against the dataset"""
    # Test parameters (fiducial ΛCDM values)
    test_H0 = 73.0
    test_Om = 0.3
    test_Ok = 0.0
    
    # Calculate predictions for all redshifts
    mu_pred = distance_modulus(test_H0, test_Om, test_Ok, z)
    
    # Calculate residuals
    residuals = mu_obs - mu_pred
    
    # Print statistics
    print("Cosmological Test Results:")
    print("-" * 50)
    print(f"Test parameters: H0={test_H0}, Om={test_Om}, Ok={test_Ok}")
    print("\nData Statistics:")
    print(f"Number of data points: {len(z)}")
    print(f"Redshift range: [{z.min():.3f}, {z.max():.3f}]")
    print(f"Observed mu range: [{mu_obs.min():.3f}, {mu_obs.max():.3f}]")
    print(f"Predicted mu range: [{mu_pred.min():.3f}, {mu_pred.max():.3f}]")
    print("\nResidual Statistics:")
    print(f"Mean residual: {jnp.mean(residuals):.3f}")
    print(f"Std residual: {jnp.std(residuals):.3f}")
    print(f"Min residual: {jnp.min(residuals):.3f}")
    print(f"Max residual: {jnp.max(residuals):.3f}")
    
    # Test a few specific points
    print("\nDetailed test points:")
    test_indices = [0, len(z)//2, -1]  # First, middle, and last points
    for idx in test_indices:
        print(f"\nPoint {idx}:")
        print(f"z = {z[idx]:.3f}")
        print(f"Predicted μ = {mu_pred[idx]:.3f}")
        print(f"Observed μ = {mu_obs[idx]:.3f}")
        print(f"Difference = {residuals[idx]:.3f}")
        print(f"Reported error = {mu_err[idx]:.3f}")
    
    # Plot residuals vs redshift
    plt.figure(figsize=(10, 6))
    plt.errorbar(z, residuals, yerr=mu_err, fmt='o', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Residual (observed - predicted)')
    plt.title('Residuals of distance modulus vs Redshift')
    plt.grid(True)
    plt.savefig(f'/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/{title}_residuals.png')
    
    # Plot Hubble diagram
    plt.figure(figsize=(10, 6))
    plt.errorbar(z, mu_obs, yerr=mu_err, fmt='o', alpha=0.5, label='Data')
    plt.plot(z, mu_pred, 'r-', label='Model')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus (μ)')
    plt.title('Distance modulus vs Redshift')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/{title}_hubble_diagram.png')


# -------------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------------

def model(z: jnp.array, mu_obs: jnp.array, mu_err: jnp.array):
    """Model for numpyro inference"""
    # Use uniform priors with reasonable ranges
    H0 = sample("H0", dist.Uniform(60.0, 80.0))  # Centered around 70 km/s/Mpc
    Om = sample("Om", dist.Uniform(0.1, 0.7))    # Centered around 0.3
    Ok = 0.0  # Fix Ok=0 for flat universe
        
    # Simple Normal likelihood
    likelihood = dist.MultivariateNormal(distance_modulus(H0, Om, Ok, z), sys_stat_cov_matrix)
    sample("obs", likelihood, obs=mu_obs)


# MCMC settings
kernel = NUTS(model, 
             target_accept_prob=0.8,
             max_tree_depth=5)

mcmc = MCMC(kernel, 
           num_warmup=10000,
           num_samples=20000,
           num_chains=4,
           chain_method='parallel',
           progress_bar=True)

# Run MCMC with explicit random key
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, z=z, mu_obs=mu_obs, mu_err=mu_err)
# After running MCMC
print(mcmc.print_summary())
samples = mcmc.get_samples()

# Plot the results
plot_samples(samples)

# Run the test
# get mean values from the samples
H0_mean = jnp.mean(samples['H0'])
Om_mean = jnp.mean(samples['Om'])
test_cosmological_calculations(H0_mean, Om_mean, 0.0, 'flat_model')





