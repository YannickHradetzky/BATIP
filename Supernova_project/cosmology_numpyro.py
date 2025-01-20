import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from functools import partial
import sympy as sp
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
from Config import Config, z_obs, mu_obs, cov_matrix, mu_err

# Force CPU for better numerical stability with large matrices
jax.config.update("jax_platform_name", "cpu")
# Enable x64 for better numerical precision
jax.config.update("jax_enable_x64", True)

class CosmologicalNumpyroInference:
    def __init__(self, z_obs, mu_obs, mu_err, cov_matrix=None, model_type="flat"):
        """Initialize the inference model with observed data"""
        self.z_obs = jnp.array(z_obs)
        self.mu_obs = jnp.array(mu_obs)
        self.mu_err = jnp.array(mu_err)
        if cov_matrix is not None:
            # Convert covariance matrix to float32 for better compatibility
            self.cov_matrix = jnp.array(cov_matrix, dtype=jnp.float32)
            # Add a small diagonal term for numerical stability
            self.cov_matrix = self.cov_matrix + jnp.eye(len(cov_matrix)) * 1e-6
        else:
            self.cov_matrix = None
        self.model_type = model_type
        
        # Setup Taylor series for curved model
        if self.model_type == "curved":
            self.Ok_sym, self.chi_sym = sp.symbols('Ok chi')
            self.setup_taylor_series(order=4)
    
    def setup_taylor_series(self, order=6):
        """Setup the Taylor series for the curvature correction"""
        if self.model_type == "curved":
            # Define the series expressions
            self._f_positive = (1 / sp.sqrt(self.Ok_sym)) * sp.sinh(sp.sqrt(self.Ok_sym) * self.chi_sym)
            self._f_negative = (1 / sp.sqrt(-self.Ok_sym)) * sp.sin(sp.sqrt(-self.Ok_sym) * self.chi_sym)
            
            # Create lambdified functions for fast numerical evaluation
            self._eval_positive = sp.lambdify((self.Ok_sym, self.chi_sym), 
                                            self._f_positive.series(self.Ok_sym, 0, order).removeO(),
                                            modules=['numpy'])
            self._eval_negative = sp.lambdify((self.Ok_sym, self.chi_sym), 
                                            self._f_negative.series(self.Ok_sym, 0, order).removeO(),
                                            modules=['numpy'])
        
    def hubble_z(self, H0, Om, Ok=None, z=None):
        """Compute Hubble parameter at redshift z"""
        z = self.z_obs if z is None else z
        matter_term = Om * (1 + z)**3
        
        if self.model_type == "flat":
            lambda_term = (1 - Om)
            return H0 * jnp.sqrt(matter_term + lambda_term)
        else:
            lambda_term = (1 - Om - Ok)
            curvature_term = Ok * (1 + z)**2
            return H0 * jnp.sqrt(matter_term + curvature_term + lambda_term)
    
    def apply_curvature_correction(self, chi, Ok):
        """Apply curvature correction using Taylor series"""
        # Convert to numpy for sympy evaluation
        chi_np = jnp.asarray(chi)
        Ok_np = jnp.asarray(Ok)
        
        # Handle positive Ok values
        pos_mask = Ok >= 0
        result = jnp.zeros_like(chi)
        
        if jnp.any(pos_mask):
            pos_chi = chi_np[pos_mask]
            pos_Ok = Ok_np[pos_mask]
            pos_result = self._eval_positive(pos_Ok, pos_chi)
            result = result.at[pos_mask].set(pos_result)
        
        # Handle negative Ok values
        neg_mask = ~pos_mask
        if jnp.any(neg_mask):
            neg_chi = chi_np[neg_mask]
            neg_Ok = Ok_np[neg_mask]
            neg_result = self._eval_negative(neg_Ok, neg_chi)
            result = result.at[neg_mask].set(neg_result)
        
        return result
    
    def luminosity_distance(self, H0, Om, Ok=None):
        """Compute luminosity distance"""
        # Create fine integration grid
        z_max = jnp.max(self.z_obs)
        n_points = 3000
        z_grid = jnp.linspace(0.001, z_max, n_points)
        
        # Calculate Hubble parameter on grid
        Hz = self.hubble_z(H0, Om, Ok, z_grid)
        
        # Compute integrand
        integrand = Config.C / Hz
        
        # Compute chi for each observed redshift
        def chi(self, h0, Om, Ok, z):
            """Comoving distance"""
            N = 1000
            z_max = jnp.max(z)
            z_min = jnp.min(z)
            z_array = jnp.linspace(z_min, z_max, N)
            dz = jnp.diff(z_array, append=z_array[-1])
            
            integrand_values = Config.C / self.hubble_z(Hz, Om, Ok, z_array)
            return jax.vmap(lambda z_point: jnp.sum(integrand_values * dz * (z_array <= z_point)))(z)
        
        chi = jax.vmap(chi)(self.H0, self.Om, self.Ok, self.z_obs)
        
        if self.model_type == "curved":
            chi = self.apply_curvature_correction(chi, Ok)
        
        return (1 + self.z_obs) * chi
    
    def distance_modulus(self, H0, Om, Ok=None):
        """Compute distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok)
        return 5 * jnp.log10(d_L + Config.EPSILON) + 25
    
    def model(self):
        """Define the probabilistic model"""
        # Sample from priors
        if self.model_type == "flat":
            H0 = numpyro.sample(
                "H0",
                dist.TruncatedNormal(
                    Config.PARAM_PRIORS['flat']['H0'][0],
                    Config.PARAM_PRIORS['flat']['H0'][1],
                    low=50,
                    high=100
                )
            )
            Om = numpyro.sample(
                "Om",
                dist.TruncatedNormal(
                    Config.PARAM_PRIORS['flat']['Om'][0],
                    Config.PARAM_PRIORS['flat']['Om'][1],
                    low=0,
                    high=1
                )
            )
            # Compute expected distance moduli
            mu_expected = self.distance_modulus(H0, Om)
        else:
            H0 = numpyro.sample(
                "H0",
                dist.TruncatedNormal(
                    Config.PARAM_PRIORS['curved']['H0'][0],
                    Config.PARAM_PRIORS['curved']['H0'][1],
                    low=50,
                    high=100
                )
            )
            Om = numpyro.sample(
                "Om",
                dist.TruncatedNormal(
                    Config.PARAM_PRIORS['curved']['Om'][0],
                    Config.PARAM_PRIORS['curved']['Om'][1],
                    low=0,
                    high=1
                )
            )
            Ok = numpyro.sample(
                "Ok",
                dist.TruncatedNormal(
                    Config.PARAM_PRIORS['curved']['Ok'][0],
                    Config.PARAM_PRIORS['curved']['Ok'][1],
                    low=-1,
                    high=1
                )
            )
            # Compute expected distance moduli
            mu_expected = self.distance_modulus(H0, Om, Ok)
        
        # Sample observations
        if self.cov_matrix is not None:
            numpyro.sample(
                "obs",
                dist.MultivariateNormal(mu_expected, self.cov_matrix),
                obs=self.mu_obs
            )
        else:
            numpyro.sample(
                "obs",
                dist.Normal(mu_expected, self.mu_err),
                obs=self.mu_obs
            )
    
    def run_mcmc(self, num_warmup=1000, num_samples=1000, num_chains=4):
        """Run MCMC inference"""
        # Initialize random number generator
        rng_key = jax.random.PRNGKey(0)
        
        # Initialize the NUTS kernel with a dense mass matrix for better sampling
        kernel = NUTS(self.model, dense_mass=True, target_accept_prob=0.8)
        
        # Run multiple chains in parallel
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="parallel",
            progress_bar=True
        )
        
        # Run inference
        mcmc.run(rng_key)
        
        # Print summary
        mcmc.print_summary()
        
        return mcmc

    def plot_results(self, mcmc, save_path=None):
        """Plot the MCMC results"""
        # Convert to arviz InferenceData
        data = az.from_numpyro(mcmc)
        
        # Create corner plot
        if self.model_type == "flat":
            var_names = ["H0", "Om"]
        else:
            var_names = ["H0", "Om", "Ok"]
        
        az.plot_pair(
            data,
            var_names=var_names,
            kind="kde",
            marginals=True,
            point_estimate="mean"
        )
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def plot_model_comparison(flat_mcmc, curved_mcmc, save_path=None):
    """Plot comparison between flat and curved models"""
    # Convert to arviz InferenceData
    flat_data = az.from_numpyro(flat_mcmc)
    curved_data = az.from_numpyro(curved_mcmc)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot H0 comparison
    az.plot_density(
        [flat_data, curved_data],
        var_names=["H0"],
        ax=axes[0],
        labels=["Flat", "Curved"]
    )
    axes[0].set_title("H0 Posterior Distribution")
    
    # Plot Om comparison
    az.plot_density(
        [flat_data, curved_data],
        var_names=["Om"],
        ax=axes[1],
        labels=["Flat", "Curved"]
    )
    axes[1].set_title("Om Posterior Distribution")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Run inference for both flat and curved models
    results = {}
    for model_type in ["flat", "curved"]:
        print(f"\nRunning inference for {model_type} model...")
        inference = CosmologicalNumpyroInference(
            z_obs=z_obs,
            mu_obs=mu_obs,
            mu_err=mu_err,
            cov_matrix=cov_matrix,
            model_type=model_type
        )
        mcmc = inference.run_mcmc()
        results[model_type] = mcmc
        
        # Plot individual results
        inference.plot_results(
            mcmc,
            save_path=f"Plots/numpyro_{model_type}_posteriors.png"
        )
    
    # Plot model comparison
    plot_model_comparison(
        results["flat"],
        results["curved"],
        save_path="Plots/numpyro_model_comparison.png"
    ) 