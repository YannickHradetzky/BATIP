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
from scipy.interpolate import approximate_taylor_polynomial
import arviz as az

# Configuration
class Config:
    # MCMC settings
    NUM_WARMUP = 300
    NUM_SAMPLES = 300
    NUM_CHAINS = 4
    TARGET_ACCEPT_PROB = 0.8
    MAX_TREE_DEPTH = 5
    
    # Priors
    H0_PRIOR = (60.0, 80.0)  # Uniform prior range for H0
    OM_PRIOR = (0.0, 0.5)    # Uniform prior range for Omega_m
    OK_PRIOR = (-0.1, 0.1)   # Uniform prior range for Omega_k
    #OK_PRIOR_MEAN = 0.0      # Mean for Ok normal prior
    #OK_PRIOR_STD = 0.1       # Std for Ok normal prior
    
    # Data filtering
    MIN_REDSHIFT = 0.000
    
    # Physics constants
    C = 299792.458  # Speed of light in km/s

    # Numerical offsets
    EPSILON = 1e-10
    
    # Paths
    DATA_PATH = '/Users/Maxi/Desktop/Uni/Master/Cosmos/Prob/Data/'
    PLOT_PATH = '/Users/Maxi/Desktop/Uni/Master/Cosmos/Prob/Plots/'

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
        
        # store the covariance matrix as a txt file
        np.savetxt(f'{Config.DATA_PATH}{title}_cov_matrix.txt', cov_matrix)
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
        z_min = jnp.min(z)
        z_max = jnp.max(z)
        z_array = jnp.linspace(z_min, z_max, N)
        dz = jnp.diff(z_array, append=z_array[-1])
        
        integrand_values = Config.C / self.hubble_z(h0, Om, Ok, z_array)
        return jax.vmap(lambda z_point: jnp.sum(integrand_values * dz * (z_array <= z_point)))(z)
    
    def r_exact(self, Ok, chi):
        """Exact radial distance function"""
        # Flat universe (Ok = 0)
        flat_case = chi
        
        # Open universe (Ok > 0)
        arg_pos = jnp.sqrt(Ok + Config.EPSILON) * chi / Config.C
        #sqrt_pos = approximate_taylor_polynomial(jnp.sinh, 0, degree=5, scale=1)(arg_pos)
        sqrt_pos = self.taylor_sinh(arg_pos)
        open_case = Config.C / jnp.sqrt(Ok + Config.EPSILON) * sqrt_pos
        
        # Closed universe (Ok < 0)
        arg_neg = jnp.sqrt(jnp.abs(Ok) + Config.EPSILON) * chi / Config.C
        #sqrt_neg = approximate_taylor_polynomial(jnp.sin, 0, degree=5, scale=1)(arg_neg)
        sqrt_neg = self.taylor_sin(arg_neg)
        closed_case = Config.C / jnp.sqrt(jnp.abs(Ok) + Config.EPSILON) * sqrt_neg
        
        return jnp.where(Ok == 0, flat_case,
                        jnp.where(Ok > 0, open_case, closed_case))
    
    def taylor_sinh(self, x):
        """Taylor series expansion for sinh around 0"""
        #return x + (x**3) / 6 + (x**5) / 120
        return jnp.sinh(x)
    
    def taylor_sin(self, x):
        """Taylor series expansion for sin around 0"""
        #return x - (x**3) / 6 + (x**5) / 120
        return jnp.sin(x)
    
    def luminosity_distance(self, H0, Om, Ok, z):
        """Luminosity distance"""
        chi_val = self.chi(H0, Om, Ok, z)
        r_z = jnp.where(jnp.abs(Ok) < Config.EPSILON, chi_val, self.r_exact(Ok, chi_val))
        #r_z = self.r_exact(Ok, chi_val)
        return (1 + z) * r_z
    
    def distance_modulus(self, H0, Om, Ok, z):
        """Distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok, z)
        return 5 * jnp.log10(d_L + Config.EPSILON) + 25
    
    def model_flat(self, z, mu_obs, mu_err):
        """Flat ΛCDM model"""
        H0 = sample("H0", dist.Uniform(*Config.H0_PRIOR))
        Om = sample("Om", dist.Uniform(*Config.OM_PRIOR))
        Ok = 0.0
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z), 
            self.sys_stat_cov_matrix
        )
        sample("obs", likelihood, obs=mu_obs)
    
    def model_curved(self, z, mu_obs, mu_err):
        """Curved ΛCDM model"""
        H0 = sample("H0", dist.Uniform(*Config.H0_PRIOR))
        Om = sample("Om", dist.Uniform(*Config.OM_PRIOR))
        Ok = sample("Ok", dist.Uniform(*Config.OM_PRIOR))
        
        likelihood = dist.MultivariateNormal(
            self.distance_modulus(H0, Om, Ok, z), 
            self.sys_stat_cov_matrix
        )
        likelihood = dist.Normal(self.distance_modulus(H0, Om, Ok, z), mu_err)
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
        
        samples = mcmc.get_samples()
        
        # Check for divergences
        #divergences = mcmc.get_extra_fields()['diverging']
        if model_type == "flat":
            print(f"Number of samples H0, Om: {len(samples['H0'])}, {len(samples['Om'])}")
        else:
            print(f"Number of samples H0, Om, Ok: {len(samples['H0'])}, {len(samples['Om'])}, {len(samples['Ok'])}")
        
        #print(f"Number of divergences: {jnp.sum(divergences)}")
        # Plot trace and pair plots
        az.plot_trace(samples)
        plt.savefig(Config.PLOT_PATH + 'trace_plot.png')
        az.plot_pair(samples)
        plt.savefig(Config.PLOT_PATH + 'pair_plot.png')
        
        # Call plot_training_data function
        theta = jnp.column_stack((samples['H0'], samples['Om'], samples['Ok'] if model_type == "curved" else jnp.zeros_like(samples['H0'])))
        simulated_data = jax.vmap(lambda h0, om, ok: self.distance_modulus(h0, om, ok, self.z))(samples['H0'], samples['Om'], samples['Ok'] if model_type == "curved" else jnp.zeros_like(samples['H0']))
        self.plot_training_data(self.z, self.mu_obs, self.mu_err, theta, simulated_data, model_type=model_type)
        
        # Highlight divergent samples
        #divergent_samples = {k: v[divergences] for k, v in samples.items()}
        #print("Divergent samples:", divergent_samples)
        
        
        return samples
    
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
    
    def plot_training_data(self, z_obs, mu_obs, mu_err, theta, simulated_data, model_type="flat"):
        """Create visualization of parameter space coverage and model predictions during training"""
        if model_type == "flat":
            fig = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(2, 2, height_ratios=[1, 0.5])
            
            # Plot parameter space coverage (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            scatter = ax1.scatter(theta[:, 0], theta[:, 1], c=simulated_data.mean(axis=1), 
                                cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ax=ax1, label='Mean μ')
            ax1.set_xlabel('H₀ [km/s/Mpc]')
            ax1.set_ylabel('Ωₘ')
            ax1.set_title('Parameter Space Coverage')
            
            # Plot model predictions vs data (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.errorbar(z_obs, mu_obs, yerr=mu_err,
                        fmt='r.', alpha=0.5, label='Data', markersize=2)
            
            # Plot all simulations to show distribution
            simulated_mean = simulated_data.mean(axis=0)
            simulated_std = simulated_data.std(axis=0)
            
            # Plot mean and standard deviation bands
            ax2.fill_between(z_obs, 
                            simulated_mean - 2*simulated_std,
                            simulated_mean + 2*simulated_std,
                            color='b', alpha=0.1, label='2σ region')
            ax2.fill_between(z_obs,
                            simulated_mean - simulated_std,
                            simulated_mean + simulated_std,
                            color='b', alpha=0.2, label='1σ region')
            ax2.plot(z_obs, simulated_mean, 'b-', label='Mean prediction')
            
            ax2.set_xlabel('Redshift (z)')
            ax2.set_ylabel('Distance Modulus (μ)')
            ax2.set_title('Model Predictions (n=50) vs Data')
            ax2.legend()
            
            # Plot parameter distributions (bottom)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(theta[:, 0], bins=50, density=True)
            ax3.set_xlabel('H₀ [km/s/Mpc]')
            ax3.set_ylabel('Density')
            ax3.set_title('H₀ Distribution')
            
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(theta[:, 1], bins=50, density=True)
            ax4.set_xlabel('Ωₘ')
            ax4.set_ylabel('Density')
            ax4.set_title('Ωₘ Distribution')
        
        else:  # curved model
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
            
            # Plot H0 vs Om parameter space (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            scatter = ax1.scatter(theta[:, 0], theta[:, 1], c=simulated_data.mean(axis=1), 
                                cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ax=ax1, label='Mean μ')
            ax1.set_xlabel('H₀ [km/s/Mpc]')
            ax1.set_ylabel('Ωₘ')
            ax1.set_title('H₀-Ωₘ Parameter Space')
            
            # Plot Om vs Ok parameter space (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            scatter = ax2.scatter(theta[:, 1], theta[:, 2], c=simulated_data.mean(axis=1), 
                                cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, ax=ax2, label='Mean μ')
            ax2.set_xlabel('Ωₘ')
            ax2.set_ylabel('Ωₖ')
            ax2.set_title('Ωₘ-Ωₖ Parameter Space')
            
            # Plot model predictions vs data (middle)
            ax3 = fig.add_subplot(gs[1, :])
            ax3.errorbar(z_obs, mu_obs, yerr=mu_err,
                        fmt='r.', alpha=0.5, label='Data', markersize=2)
            
            # Plot all simulations to show distribution
            simulated_mean = simulated_data.mean(axis=0)
            simulated_std = simulated_data.std(axis=0)
            
            # Plot mean and standard deviation bands
            ax3.fill_between(z_obs, 
                            simulated_mean - 2*simulated_std,
                            simulated_mean + 2*simulated_std,
                            color='b', alpha=0.2, label='95% CI')
            ax3.fill_between(z_obs,
                            simulated_mean - simulated_std, 
                            simulated_mean + simulated_std,
                            color='b', alpha=0.3, label='68% CI')
            ax3.plot(z_obs, simulated_mean, 'b-', label='Mean prediction')
            
            # Plot all individual simulations with high transparency
            for i in range(len(simulated_data)):
                ax3.plot(z_obs, simulated_data[i],
                        'b-', alpha=0.01)
            
            ax3.set_xlabel('Redshift (z)')
            ax3.set_ylabel('Distance Modulus (μ)')
            ax3.set_title('Model Predictions (n=50) vs Data')
            ax3.legend()
            
            # Plot parameter distributions (bottom)
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.hist(theta[:, 0], bins=50, density=True)
            ax4.set_xlabel('H₀ [km/s/Mpc]')
            ax4.set_ylabel('Density')
            ax4.set_title('H₀ Distribution')
            
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.hist(theta[:, 1], bins=50, density=True, alpha=0.5, label='Ωₘ')
            ax5.hist(theta[:, 2], bins=50, density=True, alpha=0.5, label='Ωₖ')
            ax5.set_xlabel('Parameter Value')
            ax5.set_ylabel('Density')
            ax5.set_title('Ωₘ and Ωₖ Distributions')
            ax5.legend()
        
        plt.tight_layout()
        plt.savefig(f'{Config.PLOT_PATH}sbi_training_data_{model_type}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
# Example usage
if __name__ == "__main__":
    # Initialize the inference object
    cosmo = CosmologyInference()
    
    # Run flat model
    #print("Running flat ΛCDM model...")
    #flat_samples = cosmo.run_inference(model_type="flat")
    #cosmo.plot_samples(flat_samples, model_type="flat")
    
    # Run curved model
    print("\nRunning curved ΛCDM model...")
    curved_samples = cosmo.run_inference(model_type="curved")
    cosmo.plot_samples(curved_samples, model_type="curved") 