import sympy as sp
import torch
from torch.distributions import MultivariateNormal, Independent, Normal
from SBIConfig import SBIConfig, z_obs, mu_obs, cov_matrix, mu_err
from SBIConfig import plot_training_data, plot_scientific_results
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
        self.setup_taylor_series(order=4)
        
    def setup_prior(self):
        """Setup the prior distribution for parameters"""
        if self.model_type == "flat":
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([SBIConfig.PARAM_PRIORS['flat']['H0'][0],
                                    SBIConfig.PARAM_PRIORS['flat']['Om'][0]], dtype=torch.float64),
                    scale=torch.tensor([SBIConfig.PARAM_PRIORS['flat']['H0'][1],
                                    SBIConfig.PARAM_PRIORS['flat']['Om'][1]], dtype=torch.float64)
                ),
                1
            )
        else:
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([SBIConfig.PARAM_PRIORS['curved']['H0'][0],
                                    SBIConfig.PARAM_PRIORS['curved']['Om'][0],
                                    SBIConfig.PARAM_PRIORS['curved']['Ok'][0]], dtype=torch.float64),
                    scale=torch.tensor([SBIConfig.PARAM_PRIORS['curved']['H0'][1],
                                    SBIConfig.PARAM_PRIORS['curved']['Om'][1],
                                    SBIConfig.PARAM_PRIORS['curved']['Ok'][1]], dtype=torch.float64)
                ),
                1
            )

    def setup_taylor_series(self, order=6):
        """Setup the Taylor series for the curvature correction"""
        f_positive = (1 / sp.sqrt(self.Ok_sym)) * sp.sinh(sp.sqrt(self.Ok_sym) * self.chi_sym)
        f_negative = (1 / sp.sqrt(-self.Ok_sym)) * sp.sin(sp.sqrt(-self.Ok_sym) * self.chi_sym)
        
        if self.model_type == "curved":
            # Compute Taylor series
            self._taylor_series_negative = sp.series(f_positive, self.Ok_sym, 0, order).removeO()
            self._taylor_series_positive = sp.series(f_negative, self.Ok_sym, 0, order).removeO()
            
            # Convert to polynomial coefficients
            self._poly_coeffs_pos = {}
            self._poly_coeffs_neg = {}
            
            # Convert positive series
            poly = sp.Poly(self._taylor_series_positive, self.Ok_sym, self.chi_sym)
            for powers, coeff in poly.terms():
                self._poly_coeffs_pos[powers] = float(coeff)
            
            # Convert negative series
            poly = sp.Poly(self._taylor_series_negative, self.Ok_sym, self.chi_sym)
            for powers, coeff in poly.terms():
                self._poly_coeffs_neg[powers] = float(coeff)

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
        integrand = SBIConfig.C / Hz
        
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
        return 5 * torch.log10(d_L + SBIConfig.EPSILON) + 25
    
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
        Apply curvature correction to chi using vectorized polynomial evaluation.
        Args:
            chi: tensor of shape (batch_size,)
            Ok: tensor of shape (batch_size,)
        Returns:
            tensor of shape (batch_size,)
        """
        # For extremely small Ok values, return chi to avoid numerical instability
        small_ok_mask = torch.abs(Ok) < 1e-5
        result = torch.zeros_like(chi)
        result[small_ok_mask] = chi[small_ok_mask]
        
        # For remaining values, use Taylor series
        remaining_mask = ~small_ok_mask
        if torch.any(remaining_mask):
            Ok_remaining = Ok[remaining_mask]
            chi_remaining = chi[remaining_mask]
            
            pos_mask = Ok_remaining >= 0
            neg_mask = ~pos_mask
            
            temp_result = torch.zeros_like(chi_remaining)
            
            # Evaluate polynomial for positive Ok values
            if torch.any(pos_mask):
                for (ok_power, chi_power), coeff in self._poly_coeffs_pos.items():
                    ok_term = Ok_remaining[pos_mask]**ok_power
                    chi_term = chi_remaining[pos_mask]**chi_power
                    temp_result[pos_mask] += coeff * ok_term * chi_term
            
            # Evaluate polynomial for negative Ok values
            if torch.any(neg_mask):
                for (ok_power, chi_power), coeff in self._poly_coeffs_neg.items():
                    ok_term = Ok_remaining[neg_mask]**ok_power
                    chi_term = chi_remaining[neg_mask]**chi_power
                    temp_result[neg_mask] += coeff * ok_term * chi_term
            
            result[remaining_mask] = temp_result
        
        return result
    
    def train(self):
        """Train the neural posterior estimator"""
        self.posterior_estimator = inference.SNPE(
            prior=self.prior,
            density_estimator="maf",
            show_progress_bars=True,
        )
        
        # Sample from prior and simulate
        theta = self.prior.sample((SBIConfig.NUM_SIMULATIONS,))
        x = self.simulate(theta, self.cov_matrix)
        
        # Convert to float32 for SBI
        theta = theta.to(torch.float32)
        x = x.to(torch.float32)[:,:,1]

        plot_training_data(self.z_obs, self.mu_obs, self.mu_err, theta, x, model_type=self.model_type)
        
        self.posterior_estimator.append_simulations(theta, x)
        density_estimator = self.posterior_estimator.train(
            training_batch_size=SBIConfig.BATCH_SIZE,
            max_num_epochs=1000,
            stop_after_epochs=50,
        )
        self.posterior = self.posterior_estimator.build_posterior(density_estimator)

    def sample_posterior(self, num_samples=None):
        """Sample from the posterior distribution"""
        if num_samples is None:
            num_samples = SBIConfig.NUM_POSTERIOR_SAMPLES
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