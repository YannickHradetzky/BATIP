import sympy as sp
import torch
from torch.distributions import MultivariateNormal, Independent, Normal
from Config import Config, z_obs, mu_obs, cov_matrix, mu_err
from Config import plot_training_data, plot_scientific_results, plot_model_comparison
import matplotlib.pyplot as plt
from sbi import inference
# set number of cores to use
torch.set_num_threads(10)

class CosmologicalSimulatorInference:
    def __init__(self, z_obs, mu_obs, mu_err, cov_matrix = None, model_type="flat"):
        """Initialize the simulator with observed redshift values"""
        # Convert all input tensors to float64
        self.z_obs = torch.as_tensor(z_obs, dtype=torch.float64).clone().detach()
        self.mu_obs = torch.as_tensor(mu_obs, dtype=torch.float64).clone().detach()
        self.mu_err = torch.as_tensor(mu_err, dtype=torch.float64).clone().detach()
        self.model_type = model_type
        self.cov_matrix = torch.as_tensor(cov_matrix, dtype=torch.float64) if cov_matrix is not None else None
        self.prior = None
        self.setup_prior()
        self.Ok_sym, self.chi_sym = sp.symbols('Ok chi')
        self.setup_taylor_series(order=8)
        
    def setup_prior(self):
        """Setup the prior distribution for parameters"""
        if self.model_type == "flat":
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([Config.PARAM_PRIORS['flat']['H0'][0],
                                    Config.PARAM_PRIORS['flat']['Om'][0]], dtype=torch.float64),
                    scale=torch.tensor([Config.PARAM_PRIORS['flat']['H0'][1],
                                    Config.PARAM_PRIORS['flat']['Om'][1]], dtype=torch.float64)
                ),
                1
            )
        else:
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([Config.PARAM_PRIORS['curved']['H0'][0],
                                    Config.PARAM_PRIORS['curved']['Om'][0],
                                    Config.PARAM_PRIORS['curved']['Ok'][0]], dtype=torch.float64),
                    scale=torch.tensor([Config.PARAM_PRIORS['curved']['H0'][1],
                                    Config.PARAM_PRIORS['curved']['Om'][1],
                                    Config.PARAM_PRIORS['curved']['Ok'][1]], dtype=torch.float64)
                ),
                1
            )

    def setup_taylor_series(self, order=6):
        """Setup the Taylor series for the curvature correction"""
        # Define the series expressions but don't expand them yet
        if self.model_type == "curved":
            self._f_positive = (1 / sp.sqrt(self.Ok_sym)) * sp.sinh(sp.sqrt(self.Ok_sym) * self.chi_sym)
            self._f_negative = (1 / sp.sqrt(-self.Ok_sym)) * sp.sin(sp.sqrt(-self.Ok_sym) * self.chi_sym)
            
            # Create lambdified functions for fast numerical evaluation
            self._eval_positive = sp.lambdify((self.Ok_sym, self.chi_sym), 
                                            self._f_positive.series(self.Ok_sym, 0, order).removeO(),
                                            modules=['numpy'])
            self._eval_negative = sp.lambdify((self.Ok_sym, self.chi_sym), 
                                            self._f_negative.series(self.Ok_sym, 0, order).removeO(),
                                            modules=['numpy'])

    def hubble_z(self, H0, Om, Ok=None):
        """Hubble parameter at redshift z"""
        # Reshape parameters to allow broadcasting with z
        H0 = H0.reshape(-1, 1)  # Shape: (batch_size, 1)
        Om = Om.reshape(-1, 1)  # Shape: (batch_size, 1)
        
        matter_term = Om * (1 + self.z_obs)**3
        if self.model_type == "flat":
            lambda_term = (1 - Om)
            return H0 * torch.sqrt(matter_term + lambda_term)
        else:
            Ok = Ok.reshape(-1, 1)  # Shape: (batch_size, 1)
            lambda_term = (1 - Om - Ok)
            curvature_term = Ok * (1 + self.z_obs)**2
            return H0 * torch.sqrt(matter_term + curvature_term + lambda_term)
    
    def luminosity_distance(self, H0, Om, Ok=None):
        """
        Compute luminosity distance with improved precision.
        Returns tensor of shape (batch_size, n_redshifts)
        """
        # Initialize output tensor
        d_L = torch.zeros((len(H0), len(self.z_obs)))
        
        # Create fine integration grid
        z_max = torch.max(self.z_obs)
        n_points = 3000
        z_grid = torch.linspace(0.001, z_max, n_points)
        
        # Reshape parameters for broadcasting
        H0_exp = H0.reshape(-1, 1)
        Om_exp = Om.reshape(-1, 1)
        
        # Calculate Hubble parameter
        matter_term = Om_exp * (1 + z_grid)**3
        if self.model_type == "flat":
            Hz = H0_exp * torch.sqrt(matter_term + (1 - Om_exp))
        else:
            Ok_exp = Ok.reshape(-1, 1)
            Hz = H0_exp * torch.sqrt(
                matter_term + 
                Ok_exp * (1 + z_grid)**2 + 
                (1 - Om_exp - Ok_exp)
            )
        
        # Debug Hz values
        if torch.any(torch.isnan(Hz)) or torch.any(torch.isinf(Hz)):
            print("Found inf/nan in Hz calculation")
            print(f"Hz range: {torch.min(Hz).item():.2f} to {torch.max(Hz).item():.2f}")

        # Compute integrand
        integrand = Config.C / Hz
        
        # Debug integrand values
        if torch.any(torch.isnan(integrand)) or torch.any(torch.isinf(integrand)):
            print("Found inf/nan in integrand")
            print(f"integrand range: {torch.min(integrand).item():.2f} to {torch.max(integrand).item():.2f}")

        # Compute luminosity distance for each redshift
        for i, z in enumerate(self.z_obs):
            z_val = float(z)
            mask = z_grid <= z_val
            chi = torch.trapz(integrand[:, mask], z_grid[mask], dim=1)
            
            if self.model_type == "curved":
                chi = self.apply_curvature_correction(chi, Ok.reshape(-1))
            
            # Debug chi values
            if torch.any(torch.isnan(chi)) or torch.any(torch.isinf(chi)):
                print(f"Found inf/nan in chi at z = {z_val:.2f}")
                print(f"chi range: {torch.min(chi).item():.2f} to {torch.max(chi).item():.2f}")
            
            d_L[:, i] = (1 + z_val) * chi
        
        # Final check on d_L
        if torch.any(torch.isnan(d_L)) or torch.any(torch.isinf(d_L)):
            print("Found inf/nan in final d_L")
            valid_mask = ~(torch.isnan(d_L) | torch.isinf(d_L))
            print(f"d_L valid range: {torch.min(d_L[valid_mask]).item():.2f} to {torch.max(d_L[valid_mask]).item():.2f}")
            print(f"First occurrence of inf/nan at z = {self.z_obs[torch.where(~valid_mask)[1][0]].item():.2f}")

        return d_L
    
    def distance_modulus(self, H0, Om, Ok=None):
        """Compute distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok)
        return 5 * torch.log10(d_L + Config.EPSILON) + 25
    
    def simulate(self, params, cov_matrix=None):
        """Simulate distance moduli for given parameters"""
        # Convert params to float64 for internal calculations
        params = params.to(torch.float64)
        
        if params.ndim == 1:
            params = params.unsqueeze(0)
        
        if self.model_type == "flat":
            H0, Om = params[:, 0], params[:, 1]
            mu = self.distance_modulus(H0, Om)
        else:
            H0, Om, Ok = params[:, 0], params[:, 1], params[:, 2]
            mu = self.distance_modulus(H0, Om, Ok)
        
        if cov_matrix is not None:
            # Create error distribution for each simulation
            batch_size = len(params)
            errors = torch.zeros_like(mu)
            
            # Sample errors for each simulation independently
            for i in range(batch_size):
                error_dist = MultivariateNormal(
                    loc=torch.zeros_like(self.z_obs),
                    covariance_matrix=cov_matrix
                )
                errors[i] = error_dist.sample()
            
            # Add errors to simulated distance moduli
            mu = mu + errors
        
        # Return both z and mu for each simulation
        return torch.stack([self.z_obs.expand(len(params), -1), mu], dim=-1)
    
    def apply_curvature_correction(self, chi, Ok):
        """
        Apply curvature correction to chi using direct series evaluation.
        Args:
            chi: tensor of shape (batch_size,)
            Ok: tensor of shape (batch_size,)
        Returns:
            tensor of shape (batch_size,)
        """
        result = torch.zeros_like(chi)
        
        # Convert tensors to numpy for sympy evaluation
        chi_np = chi.numpy()
        Ok_np = Ok.numpy()
        
        # Handle positive Ok values
        pos_mask = Ok >= 0
        if torch.any(pos_mask):
            pos_chi = chi_np[pos_mask]
            pos_Ok = Ok_np[pos_mask]
            pos_result = self._eval_positive(pos_Ok, pos_chi)
            result[pos_mask] = torch.from_numpy(pos_result)
        
        # Handle negative Ok values
        neg_mask = ~pos_mask
        if torch.any(neg_mask):
            neg_chi = chi_np[neg_mask]
            neg_Ok = Ok_np[neg_mask]
            neg_result = self._eval_negative(neg_Ok, neg_chi)
            result[neg_mask] = torch.from_numpy(neg_result)
        
        return result
    
    def train(self):
        """Train the neural posterior estimator"""
        self.posterior_estimator = inference.SNPE(
            prior=self.prior,
            density_estimator="maf",
            show_progress_bars=True,
        )
        
        # Sample and check Ok distribution (only for curved model)
        theta = self.prior.sample((Config.NUM_SIMULATIONS,))
        if self.model_type == "curved":
            print("\nChecking Ok values:")
            print(f"Prior Ok range: {theta[:,2].min():.4f} to {theta[:,2].max():.4f}")
            print(f"Prior Ok mean: {theta[:,2].mean():.4f}")
            print(f"Number of negative Ok: {(theta[:,2] < 0).sum()}")
            print(f"Number of positive Ok: {(theta[:,2] > 0).sum()}\n")
        
        x = self.simulate(theta, self.cov_matrix) 
        
        # Convert to float32 for SBI
        theta = theta.to(torch.float32)
        # ensure that Om is between 0 and 1
        theta[:,1] = torch.clamp(theta[:,1], 0, 1)
        if self.model_type == "curved":
            # ensure that Ok is between -1 and 1
            theta[:,2] = torch.clamp(theta[:,2], -1, 1)
            # ensure that H0 is between 50 and 100
        theta[:,0] = torch.clamp(theta[:,0], 50, 100)
        x = x.to(torch.float32)[:,:,1]  # Only keep the mu values

        # Remove samples with NaN values and check Ok distribution again
        nan_mask = torch.isnan(x).any(dim=1)
        theta = theta[~nan_mask]
        x = x[~nan_mask]

        if self.model_type == "curved":
            print("After NaN filtering:")
            print(f"Ok range: {theta[:,2].min():.4f} to {theta[:,2].max():.4f}")
            print(f"Ok mean: {theta[:,2].mean():.4f}")
            print(f"Number of negative Ok: {(theta[:,2] < 0).sum()}")
            print(f"Number of positive Ok: {(theta[:,2] > 0).sum()}")
            print(f"Total samples removed: {nan_mask.sum()}\n")
        
        plot_training_data(self.z_obs, self.mu_obs, self.mu_err, theta, x, model_type=self.model_type)
        
        self.posterior_estimator.append_simulations(theta, x)
        density_estimator = self.posterior_estimator.train(
            training_batch_size=Config.BATCH_SIZE,
            max_num_epochs=1000,
            stop_after_epochs=50,
        )
        self.posterior = self.posterior_estimator.build_posterior(density_estimator)

    def sample_posterior(self, num_samples=None):
        """Sample from the posterior distribution"""
        if num_samples is None:
            num_samples = Config.NUM_POSTERIOR_SAMPLES
        return self.posterior.sample((num_samples,), x=self.mu_obs)



    
if __name__ == "__main__":
    samples_dict = {}
    for model_type in ["flat", "curved"]:
        simulator = CosmologicalSimulatorInference(
                                    z_obs,
                                    mu_obs,
                                    mu_err,
                                    model_type=model_type
        )
        simulator.train()
        samples = simulator.sample_posterior()
        samples_dict[model_type] = samples

    plot_scientific_results(samples_dict["flat"], samples_dict["curved"])
    plot_model_comparison(samples_dict["flat"], samples_dict["curved"])

    
    # # draw samples from prior
    # simulator = CosmologicalSimulatorInference(
    #     z_obs,
    #     mu_obs,
    #     mu_err,
    #     model_type="curved"
    # )
    # samples = simulator.prior.sample((100,))
    # # simulate data
    # data = simulator.simulate(samples)[:,:,1]
    # # plot real data in background
    # plt.errorbar(z_obs, mu_obs, yerr=mu_err, fmt='r-', label='Real data', zorder=1)

    # # plot simulated data on top
    # for i in range(len(data)):
    #     plt.plot(z_obs, data[i], 'b-', alpha=0.5, zorder=2)

    # plt.legend()
    # plt.show()
