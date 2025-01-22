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

# Configuration
class Config:
    # MCMC settings
    NUM_WARMUP = 1000
    NUM_SAMPLES = 2000
    NUM_CHAINS = 4
    TARGET_ACCEPT_PROB = 0.8
    MAX_TREE_DEPTH = 5
    
    # Priors
    H0_PRIOR = (60.0, 80.0)  # Uniform prior range for H0
    OM_PRIOR = (0.0, 0.9)    # Uniform prior range for Omega_m
    OK_PRIOR_MEAN = 0.0      # Mean for Ok normal prior
    OK_PRIOR_STD = 0.2       # Std for Ok normal prior
    
    # Data filtering
    MIN_REDSHIFT = 0.001
    
    # Physics constants
    C = 299792.458  # Speed of light in km/s

    # Numerical offsets
    EPSILON = 1e-10
    
    # Paths
    DATA_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/'
    PLOT_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/'

# Set up JAX and NumPyro
jax.config.update('jax_platform_name', 'cpu')
numpyro.set_host_device_count(Config.NUM_CHAINS)

class CosmologyInference:
    def __init__(self):
        self.load_data()
        self.create_covariance_matrices()
        self.filter_data()
    
    def load_data(self):
        # Load the data
        df = pd.read_csv(Config.DATA_PATH + 'Pantheon+SH0ES.dat', sep='\s+', header=0)
        self.z = df['zCMB'].values
        self.mu_obs = df['MU_SH0ES'].values
        self.mu_err = df['MU_SH0ES_ERR_DIAG'].values
        
        # Load covariance matrices
        self.cov_data_stat = np.loadtxt(Config.DATA_PATH + 'Pantheon+SH0ES_STATONLY.cov')
        self.cov_data_sys_stat = np.loadtxt(Config.DATA_PATH + 'Pantheon+SH0ES_STAT+SYS.cov')
    
    def create_covariance_matrices(self):
        self.sys_stat_cov_matrix = self._create_cov_mat(self.cov_data_sys_stat, 'sys_and_stat')
        self.stat_cov_matrix = self._create_cov_mat(self.cov_data_stat, 'stat_only')
    
    def filter_data(self):
        print("Original data points:", len(self.z))
        mask = self.z > Config.MIN_REDSHIFT
        self.z = self.z[mask]
        self.mu_obs = self.mu_obs[mask]
        self.mu_err = self.mu_err[mask]
        print("Filtered data points:", len(self.z))
    
    @staticmethod
    def _create_cov_mat(cov_data, title):
        n = int(cov_data[0])
        cov_matrix = cov_data[1:].reshape(n, n)
        
        # Make symmetric if needed
        if not np.allclose(cov_matrix, cov_matrix.T):
            cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        
        return cov_matrix
    
    @staticmethod
    def hubble_z(H0, Om, Ok, z, flat=True):
        """Hubble parameter at redshift z"""
        matter_term = Om * (1 + z)**3
        if flat:
            lambda_term = (1 - Om)
        else:
            lambda_term = (1 - Om - Ok)
            curvature_term = Ok * (1 + z)**2
            return H0 * jnp.sqrt(matter_term + lambda_term + curvature_term)
        return H0 * jnp.sqrt(matter_term + lambda_term)
    
    def chi(self, h0, Om, Ok, z, flat=True):
        """Comoving distance"""
        N = 1000
        z_max = jnp.max(z)
        z_array = jnp.linspace(0.001, z_max, N)
        dz = jnp.diff(z_array, append=z_array[-1])
        
        integrand_values = Config.C / self.hubble_z(h0, Om, Ok, z_array, flat)
        return jax.vmap(lambda z_point: jnp.sum(integrand_values * dz * (z_array <= z_point)))(z)
    
    def luminosity_distance(self, H0, Om, Ok, z, flat=True):
        """Luminosity distance"""
        # TODO: curvature correction!!!
        chi_val = self.chi(H0, Om, Ok, z, flat)
        if flat:
            return (1 + z) * chi_val
        Ok_abs = jnp.abs(Ok)
        sqrt_Ok = jnp.sqrt(Ok_abs)
        sinh_term = jnp.sinh(sqrt_Ok * chi_val / Config.C)
        sin_term = jnp.sin(sqrt_Ok * chi_val / Config.C)
        chi_val = jnp.where(
            Ok > 0, # if open universe
            Config.C / sqrt_Ok * sinh_term, # return open universe
            Config.C / sqrt_Ok * sin_term # elsereturn closed universe
        )
        return (1 + z) * chi_val
     
    def distance_modulus(self, H0, Om, Ok, z, flat=True):
        """Distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok, z, flat)
        return 5 * jnp.log10(d_L + Config.EPSILON) + 25
    
    def model_flat(self, z, mu_obs, mu_err):
        """Flat ΛCDM model"""
        H0 = sample("H0", dist.Uniform(*Config.H0_PRIOR))
        Om = sample("Om", dist.Uniform(*Config.OM_PRIOR))
        Ok = 0.0
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z, flat=True), 
            self.sys_stat_cov_matrix
        )
        sample("obs", likelihood, obs=mu_obs)
    
    def model_curved(self, z, mu_obs, mu_err):
        """Curved ΛCDM model"""
        H0 = sample("H0", dist.Uniform(*Config.H0_PRIOR))
        Om = sample("Om", dist.Uniform(*Config.OM_PRIOR))
        Ok = sample("Ok", dist.Normal(Config.OK_PRIOR_MEAN, Config.OK_PRIOR_STD))
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z, flat=False), 
            self.sys_stat_cov_matrix
        )
        sample("obs", likelihood, obs=mu_obs)
    
    def run_inference(self, model_type="flat"):
        """Run MCMC inference"""
        model = self.model_flat if model_type == "flat" else self.model_curved
        
        kernel = NUTS(model, 
                     target_accept_prob=Config.TARGET_ACCEPT_PROB,
                     max_tree_depth=Config.MAX_TREE_DEPTH)
        
        mcmc = MCMC(kernel, 
                    num_warmup=Config.NUM_WARMUP,
                    num_samples=Config.NUM_SAMPLES,
                    num_chains=Config.NUM_CHAINS,
                    chain_method='parallel',
                    progress_bar=True)
        
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, z=self.z, mu_obs=self.mu_obs, mu_err=self.mu_err)
        
        print(mcmc.print_summary())
        return mcmc.get_samples()
    
    def plot_samples(self, samples, model_type="flat"):
        """Plot MCMC samples"""
        Omega_lambda = 1 - samples['Om'] if model_type == "flat" else 1 - samples['Om'] - samples['Ok']
        
        if model_type == "flat":
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))
        
        # H0 distribution
        axes[0].hist(samples['H0'], bins=50, color='skyblue', edgecolor='black', density=True)
        axes[0].set_title('H₀ Distribution')
        axes[0].set_xlabel('H₀ [km/s/Mpc]')
        axes[0].set_ylabel('Frequency')
        
        # Omega_m distribution
        axes[1].hist(samples['Om'], bins=50, color='lightgreen', edgecolor='black', density=True)
        axes[1].set_title('Ωₘ Distribution')
        axes[1].set_xlabel('Ωₘ')
        axes[1].set_ylabel('Frequency')
        
        # Omega_lambda distribution
        axes[2].hist(Omega_lambda, bins=50, color='salmon', edgecolor='black', density=True)
        axes[2].set_title('Ωₗ Distribution')
        axes[2].set_xlabel('Ωₗ')
        axes[2].set_ylabel('Frequency')

        if model_type == "curved":
            # Ok distribution
            axes[3].hist(samples['Ok'], bins=50, color='purple', edgecolor='black', density=True)
            axes[3].set_title('Ωₖ Distribution')
            axes[3].set_xlabel('Ωₖ')
            axes[3].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{Config.PLOT_PATH}{model_type}_posteriors.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize the inference object
    cosmo = CosmologyInference()
    
    # Run flat model
    print("Running flat ΛCDM model...")
    flat_samples = cosmo.run_inference(model_type="flat")
    cosmo.plot_samples(flat_samples, model_type="flat")
    
    # Run curved model
    print("\nRunning curved ΛCDM model...")
    curved_samples = cosmo.run_inference(model_type="curved")
    cosmo.plot_samples(curved_samples, model_type="curved") 