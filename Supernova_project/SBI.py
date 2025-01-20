import torch 
from torch.distributions import Independent, Uniform
import numpy as np
import matplotlib.pyplot as plt
from sbi import inference
from Config import load_real_data
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm

# tell torch to use more of my cpu cores
torch.set_num_threads(10)

def _simulate_single(args):
    """Helper function for parallel processing"""
    theta_single, z_obs, model = args
    Ho, Om, Ok = theta_single
    mu = torch.zeros(len(z_obs))
    
    for z_idx, z in enumerate(z_obs):
        comoving_distance = model.calc_comoving_distance(Ho, Om, Ok, z)
        luminosity_distance = model.calc_luminosity_distance(comoving_distance, Ok, z)
        mu[z_idx] = model.calc_apparent_magnitude(luminosity_distance)
    
    return mu

class physics_model:
    def __init__(self):
        self.C = 299792.458 # km/s
        self.eps = 1e-10
        self.test()

    def calc_hubble_distance(self, hubble, matter, curv, z):
        matter_term = matter * (1 + z)**3
        curvature_term = curv * (1 + z)**2
        lambda_term = 1 - matter - curv
        hubble_term = hubble * (1 + z)
        return torch.max(hubble * torch.sqrt(matter_term + curvature_term + lambda_term), 
                        torch.tensor(self.eps, dtype=torch.float64))
    
    def calc_comoving_distance(self, hubble, matter, curv, z):
        # Convert z to float if it's a tensor
        hubble_distance = self.calc_hubble_distance(hubble, matter, curv, z)
        z_values = torch.linspace(0, z, 1000)
        hubble_distance_values = self.calc_hubble_distance(hubble, matter, curv, z_values)
        return self.C * torch.trapz(1 / hubble_distance_values, z_values)
    
    def calc_luminosity_distance(self, comoving_distance, Ok, z):
        # Handle the flat universe case first
        if abs(Ok) < 1e-10:
            return (1 + z) * comoving_distance
        # For curved universes, use the exact formula
        Ok_abs = abs(Ok)
        sqrt_Ok = torch.sqrt(Ok_abs)
        if Ok > 0:
            # Open universe
            sinh_term = torch.sinh(sqrt_Ok * comoving_distance / self.C)
            transverse_distance = self.C / sqrt_Ok * sinh_term
        else:
            # Closed universe
            sin_term = torch.sin(sqrt_Ok * comoving_distance / self.C)
            transverse_distance = self.C / sqrt_Ok * sin_term
        return (1 + z) * transverse_distance
    
    def calc_apparent_magnitude(self, luminosity_distance):
        return 5 * torch.log10(luminosity_distance) + 25
    
    def test(self):
        # test for fixed parameters
        hubble = torch.tensor(70)
        matter = torch.tensor(0.3)
        curv = torch.tensor(0.1)
        z = torch.tensor(0.1)
        comoving_distance = self.calc_comoving_distance(hubble, matter, curv, z)
        luminosity_distance = self.calc_luminosity_distance(comoving_distance, curv, z)
        # real values 
        real = {
            "comoving_distance": 416.5,
            "luminosity_distance": 458.2,
        }
        # round results
        comoving_distance = round(comoving_distance.item(), 1)
        luminosity_distance = round(luminosity_distance.item(), 1)
        print(f"comoving_distance: {comoving_distance}, luminosity_distance: {luminosity_distance}")
        # check if results are close to real values
        assert abs(comoving_distance - real["comoving_distance"]) < 0.1
        assert abs(luminosity_distance - real["luminosity_distance"]) < 0.1
        print("Test passed")


class sbi_model:
    def __init__(self, num_simulations=1000):
        self.model = physics_model()
        self.prior = None
        self.Ho_bounds = (60, 80)
        self.Om_bounds = (0.1, 0.5)
        self.Ok_bounds = (-0.1, 0.1)
        self.setup_prior()
        self.z_obs, self.mu_obs, self.mu_err, self.cov_matrix = load_real_data()
        self.num_simulations = num_simulations

    def setup_prior(self):
        self.prior = Independent(Uniform(
                    low=torch.tensor([
                        self.Ho_bounds[0],
                        self.Om_bounds[0],
                        self.Ok_bounds[0],
                    ], dtype=torch.float32),
                    high=torch.tensor([
                        self.Ho_bounds[1],
                        self.Om_bounds[1],
                        self.Ok_bounds[1],
                    ], dtype=torch.float32),
                ), 1)
        
    def simulator(self, theta):
        """
        Simulate distance moduli for given parameters using parallel processing
        Input: theta shape [N_simulations, 3] (Ho, Om, Ok)
        Output: mu shape [N_simulations, len(z_obs)]
        """
        # Create arguments for each simulation
        args = [(theta[i], self.z_obs, self.model) for i in range(len(theta))]
        
        # Use torch multiprocessing to parallelize
        with Pool() as pool:
            results = list(tqdm(
                pool.imap(_simulate_single, args),
                total=len(theta),
                desc="Simulating"
            ))
        
        # Stack results into a single tensor
        return torch.stack(results)

    def train(self):
        """Train the neural posterior estimator"""
        self.posterior_estimator = inference.SNPE(
            prior=self.prior,
            density_estimator="maf",
            show_progress_bars=True,
        )
        
        # Sample from prior and convert to float32
        theta = self.prior.sample((self.num_simulations,)).to(torch.float32)
        
        # Simulate and convert to float32
        x = self.simulator(theta)
        print(f"x.shape: {x.shape}")  # Debug print
        print("Training...")
        # Append simulations and train
        self.posterior_estimator.append_simulations(theta, x)
        density_estimator = self.posterior_estimator.train(
            training_batch_size=50
        )
        self.posterior = self.posterior_estimator.build_posterior(density_estimator, sample_with="mcmc")

    def sample_posterior(self, num_samples=1000):
        """Sample from the posterior distribution"""
        # Get the observed mu value and ensure correct shape
        x_o = self.mu_obs.clone().to(torch.float32)  # Convert to float32
        print(x_o.shape)
        
        # Sample from posterior with observed data as context
        theta = self.posterior.sample((num_samples,), x=x_o)
        return theta


if __name__ == "__main__":
    model = sbi_model(num_simulations=10000)
    model.train()
    # Now sample with the actual observed data
    samples = model.sample_posterior(100000)
    # plot the samples
    Ho = samples[:, 0]
    Om = samples[:, 1]
    Ok = samples[:, 2]
    Ol = 1 - Om - Ok
    # create histograms in subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    ax1.hist(Ho, bins=50, color='blue', edgecolor='black', density=True)
    ax1.set_title('H0 Distribution')
    
    ax2.hist(Om, bins=50, color='green', edgecolor='black', density=True)
    ax2.set_title('Ωm Distribution')
    
    ax3.hist(Ok, bins=50, color='red', edgecolor='black', density=True)
    ax3.set_title('Ωk Distribution')
    
    ax4.hist(Ol, bins=50, color='purple', edgecolor='black', density=True)
    ax4.set_title('ΩΛ Distribution')
    
    plt.tight_layout()
    plt.show()
