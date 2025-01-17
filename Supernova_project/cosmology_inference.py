import numpyro.infer.initialization
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
import corner
# Configuration
class Config:
    # MCMC settings
    NUM_WARMUP = 2000
    NUM_SAMPLES = 5000
    NUM_CHAINS = 4
    TARGET_ACCEPT_PROB = 0.8
    MAX_TREE_DEPTH = 8
    
    # Priors
    H0_PRIOR = dist.Normal(70, 5)
    OM_PRIOR = dist.Normal(0.3, 0.1)
    OK_PRIOR = dist.Normal(0, 0.1)
    
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
        
        # Plot covariance matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cov_matrix, cmap='viridis')
        plt.colorbar()
        plt.savefig(f'{Config.PLOT_PATH}{title}_cov_matrix.png')
        plt.close()
        
        return cov_matrix
    
    @staticmethod
    def hubble_z(H0, Om, Ok, z):
        """Hubble parameter at redshift z"""
        matter_term = Om * (1 + z)**3
        lambda_term = jnp.where(jnp.abs(Ok) < Config.EPSILON,
                              1 - Om,  # flat universe case
                              1 - Om - Ok)  # curved universe case
        
        curvature_term = Ok * (1 + z)**2
        return H0 * jnp.sqrt(matter_term + lambda_term + jnp.where(jnp.abs(Ok) < Config.EPSILON, 0.0, curvature_term))
    
    def chi(self, h0, Om, Ok, z):
        """Comoving distance"""
        N = 1000
        z_max = jnp.max(z)
        z_min = jnp.min(z)
        z_array = jnp.linspace(z_min, z_max, N)
        dz = jnp.diff(z_array, append=z_array[-1])
        
        integrand_values = Config.C / self.hubble_z(h0, Om, Ok, z_array)
        return jax.vmap(lambda z_point: jnp.sum(integrand_values * dz * (z_array <= z_point)))(z)
    
    def r_exact(self, Ok, chi):
        """Exact radial distance function"""
        # Flat universe (Ok = 0)
        flat_case = chi
        # Open universe (Ok > 0)
        open_case = 1/jnp.sqrt(Ok + Config.EPSILON) * jnp.sinh(jnp.sqrt(Ok + Config.EPSILON) * chi)
        # Closed universe (Ok < 0)
        closed_case = 1/jnp.sqrt(jnp.abs(Ok) + Config.EPSILON) * jnp.sin(jnp.sqrt(jnp.abs(Ok)) * chi)
        
        return jnp.where(Ok == 0, flat_case,
                        jnp.where(Ok > 0, open_case, closed_case))

    def r_taylor(self, Ok, chi):
        """Taylor expansion of radial distance around Ok = 0 and differentiate between positive and negative Ok"""      
        def positive_ok(Ok, chi):
            return chi/jnp.sqrt(Ok + Config.EPSILON) + (chi**3)/6 * jnp.sqrt(Ok + Config.EPSILON) + 1/120 * (chi**5) * jnp.sqrt(Ok + Config.EPSILON)**3
        def negative_ok(Ok, chi):
            return chi/jnp.sqrt(-Ok + Config.EPSILON) - (chi**3)/6 * jnp.sqrt(-Ok + Config.EPSILON) + 1/120 * (chi**5) * jnp.sqrt(-Ok + Config.EPSILON)**3
        result = jnp.where(Ok > 0, positive_ok(Ok, chi), negative_ok(Ok, chi))
        return result

    def luminosity_distance(self, H0, Om, Ok, z):
        """Luminosity distance using Taylor expansion around Ok = 0"""
        chi_val = self.chi(H0, Om, Ok, z)
        r_z = jnp.where(jnp.abs(Ok) < Config.EPSILON, chi_val, self.r_taylor(Ok, chi_val))
        return r_z * (1 + z)

    def distance_modulus(self, H0, Om, Ok, z):
        """Distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok, z)
        return 5 * jnp.log10(d_L + Config.EPSILON) + 25
    
    def model_flat(self, z, mu_obs, mu_err):
        """Flat ΛCDM model"""
        H0 = sample("H0", Config.H0_PRIOR)
        Om = sample("Om", Config.OM_PRIOR)
        Ok = 0.0
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z), 
            self.sys_stat_cov_matrix
        )
        sample("obs", likelihood, obs=mu_obs)
    
    def model_curved(self, z, mu_obs, mu_err):
        """Curved ΛCDM model"""
        H0 = sample("H0", Config.H0_PRIOR)
        Om = sample("Om", Config.OM_PRIOR)
        Ok = sample("Ok", Config.OK_PRIOR)
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z), 
            self.sys_stat_cov_matrix
        )
        sample("obs", likelihood, obs=mu_obs)
    
    def run_inference(self, model_type="flat"):
        """Run MCMC inference"""
        model = self.model_flat if model_type == "flat" else self.model_curved
        kernel = NUTS(model, 
                     target_accept_prob=Config.TARGET_ACCEPT_PROB,
                     max_tree_depth=Config.MAX_TREE_DEPTH
        )
        
        mcmc = MCMC(kernel, 
                    num_warmup=Config.NUM_WARMUP,
                    num_samples=Config.NUM_SAMPLES,
                    num_chains=Config.NUM_CHAINS,
                    chain_method='parallel',
                    progress_bar=True
        )
        
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

    def test_model(self, model_type="flat"):
        """Test model"""
        # draw 10000 samples from the prior
        Om_prior = Config.OM_PRIOR
        Om = np.array(Om_prior.sample(random.PRNGKey(0), (10000,)))
        H0_prior = Config.H0_PRIOR
        H0 = np.array(H0_prior.sample(random.PRNGKey(0), (10000,)))
        if model_type == "curved":
            Ok_prior = Config.OK_PRIOR
            Ok = np.array(Ok_prior.sample(random.PRNGKey(0), (10000,)))
        else:
            Ok = np.zeros(10000)
        
        # Define ranges for the corner plot
        ranges = [
            (50, 80),     # H0 range
            (0.1, 0.5),   # Om range
            (-0.03, 0.03) # Ok range
        ]
        
        # make a corner plot
        fig = corner.corner(
            np.stack([H0, Om, Ok], axis=-1), 
            labels=['H0', 'Om', 'Ok'],
            range=ranges,
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            fill_contours=True,
            levels=[0.68, 0.95, 0.997],
            plot_contours_kwargs={'colors': ['red', 'green', 'blue']}
        )
        plt.savefig(f'{Config.PLOT_PATH}{model_type}_corner_plot.png')
        plt.close()

        # plot the distance modulus for each combination
        plt.figure(figsize=(10, 8))
        plt.title('Distance Modulus')
        plt.xlabel('Redshift')
        plt.ylabel('Distance Modulus')
        plt.grid(True)

        for (Om,Ho,Ok) in zip(Om, H0, Ok):
            mu = self.distance_modulus(Ho, Om, Ok, self.z)
            plt.plot(self.z, mu, label=f'H0={Ho}, Om={Om}, Ok={Ok}')
        plt.savefig(f'{Config.PLOT_PATH}{model_type}_distance_modulus.png')
        plt.close()

        print(f"Plot saved to {Config.PLOT_PATH}{model_type}_distance_modulus.png")

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
    
    # Test model
    cosmo.test_model(model_type="flat")
    cosmo.test_model(model_type="curved")