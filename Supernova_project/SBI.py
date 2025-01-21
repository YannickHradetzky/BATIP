import torch 
from torch.distributions import Independent, Uniform, MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from sbi import inference
from Config import load_real_data
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
from Plots import plot_training_data, PLOT_PATH
import corner

# set all seeds
torch.manual_seed(42)
np.random.seed(42)

# tell torch to use more of my cpu cores
torch.set_num_threads(10)

def _simulate_single(args):
    """Helper function for parallel processing"""
    theta_single, z_obs, model, cov_matrix = args
    Ho, Om, Ok = theta_single
    mu = torch.zeros(len(z_obs))
    
    # First calculate all mu values
    for z_idx, z in enumerate(z_obs):
        comoving_distance = model.calc_comoving_distance(Ho, Om, Ok, z)
        luminosity_distance = model.calc_luminosity_distance(comoving_distance, Ok, z)
        mu[z_idx] = model.calc_apparent_magnitude(luminosity_distance)
    
    # Then add noise from multivariate normal if covariance matrix is provided
    if cov_matrix is not None:
        noise = MultivariateNormal(
            loc=torch.zeros(len(z_obs)), 
            covariance_matrix=cov_matrix
        ).sample()
        mu += noise

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
        return torch.max(hubble * torch.sqrt(matter_term + curvature_term + lambda_term), 
                        torch.tensor(self.eps, dtype=torch.float64))
    
    def calc_comoving_distance(self, hubble, matter, curv, z):
        # Convert z to float if it's a tensor
        z_values = torch.linspace(0, z, 1000)
        hubble_distance_values = self.calc_hubble_distance(hubble, matter, curv, z_values)
        return self.C * torch.trapz(1 / hubble_distance_values, z_values)
    
    def calc_luminosity_distance(self, comoving_distance, Ok, z):
        # Handle the flat universe case first
        if abs(Ok) < self.eps:
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
    def __init__(self, num_simulations=1000, type="curved"):
        self.type = type
        self.model = physics_model()
        self.prior = None
        self.Ho_bounds = (60, 80)
        self.Om_bounds = (0.1, 0.5)
        self.Ok_bounds = (-0.2, 0.2)
        self.setup_prior()
        self.z_obs, self.mu_obs, self.mu_err, self.cov_matrix = load_real_data()
        self.num_simulations = num_simulations
        # turn cov_matrix into torch tensor
        self.cov_matrix = torch.tensor(self.cov_matrix, dtype=torch.float32)
        
    def setup_prior(self):
        if self.type == "flat":
            # 2D prior for flat universe (Ho, Om only)
            self.prior = Independent(Uniform(
                        low=torch.tensor([
                            self.Ho_bounds[0],
                            self.Om_bounds[0],
                        ], dtype=torch.float32),
                        high=torch.tensor([
                            self.Ho_bounds[1],
                            self.Om_bounds[1],
                        ], dtype=torch.float32),
                    ), 1)
        else:
            # 3D prior for curved universe (Ho, Om, Ok)
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
        Input: theta shape [N_simulations, 2] (Ho, Om) for flat or [N_simulations, 3] (Ho, Om, Ok) for curved
        Output: mu shape [N_simulations, len(z_obs)]
        """
        if self.type == "flat":
            # Add Ok=0 for flat universe
            theta_with_ok = torch.cat([
                theta, 
                torch.zeros(len(theta), 1, dtype=torch.float32)
            ], dim=1)
        else:
            theta_with_ok = theta

        # Create arguments for each simulation
        args = [(theta_with_ok[i], self.z_obs, self.model, self.cov_matrix) 
               for i in range(len(theta_with_ok))]
        
        # Use torch multiprocessing to parallelize
        with Pool() as pool:
            results = list(tqdm(
                pool.imap(_simulate_single, args),
                total=len(theta_with_ok),
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

        # plot_training_data(theta, self.z_obs, self.mu_obs, x)
        print(f"x.shape: {x.shape}")  # Debug print
        print("Training...")
        # Append simulations and train
        self.posterior_estimator.append_simulations(theta, x)
        density_estimator = self.posterior_estimator.train(
            training_batch_size=1000
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
    dict_storage = {}
    for type in ["curved", "flat"]:
        print(f"Training {type} model")
        model = sbi_model(num_simulations=100000, type=type)
        model.train()
        # Now sample with the actual observed data
        print(f"Sampling from {type} model")
        samples = model.sample_posterior(1000000)
        dict_storage[type] = samples
        # write samples to file
        print(f"Writing samples to file for {type} model")
        np.savetxt(PLOT_PATH + f'samples_{type}.txt', samples)
        # plot the samples
        print(f"Plotting samples for {type} model")
        Ho = samples[:, 0]
        Om = samples[:, 1]
        if type == "curved":
            Ok = samples[:, 2]
            Ol = 1 - Om - Ok
        else:
            Ol = 1 - Om
        # create histograms in subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        ax1.hist(Ho, bins=50, color='blue', edgecolor='black', density=True)
        ax1.set_title('H0 Distribution')
        
        ax2.hist(Om, bins=50, color='green', edgecolor='black', density=True)
        ax2.set_title('Ωm Distribution')
        
        if type == "curved":
            ax3.hist(Ok, bins=50, color='red', edgecolor='black', density=True)
            ax3.set_title('Ωk Distribution')
        else:
            ax3.set_visible(False)
        
        ax4.hist(Ol, bins=50, color='purple', edgecolor='black', density=True)
        ax4.set_title('ΩΛ Distribution')
        
        plt.tight_layout()
        plt.savefig(PLOT_PATH + f'parameter_distributions_{type}.png')

    # plot the distributions for each type
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"flat": "skyblue", "curved": "orange"}
    alpha = 0.7
    
    for type in ["flat", "curved"]:
        axs[0, 0].hist(dict_storage[type][:, 0], bins=50, color=colors[type], 
                       edgecolor='black', density=True, alpha=alpha, label=type)
        axs[0, 0].set_title('H0 Distribution')
        axs[0, 0].legend()
        
        axs[0, 1].hist(dict_storage[type][:, 1], bins=50, color=colors[type], 
                       edgecolor='black', density=True, alpha=alpha, label=type)
        axs[0, 1].set_title('Ωm Distribution')
        axs[0, 1].legend()

        if type == "curved":
            axs[1, 0].hist(dict_storage[type][:, 2], bins=50, color=colors[type], 
                       edgecolor='black', density=True, alpha=alpha, label=type)
            axs[1, 0].set_title('Ωk Distribution')
            axs[1, 0].legend()
        else:
            axs[1, 0].set_visible(False)
        
        if type == "curved":
            axs[1, 1].hist(1 - dict_storage[type][:, 1] - dict_storage[type][:, 2], 
                       bins=50, color=colors[type], edgecolor='black', 
                       density=True, alpha=alpha, label=type)
            axs[1, 1].set_title('ΩΛ Distribution')
            axs[1, 1].legend()
        else:
            axs[1, 1].hist(1 - dict_storage[type][:, 1], bins=50, color=colors[type], 
                       edgecolor='black', density=True, alpha=alpha, label=type)
            axs[1, 1].set_title('ΩΛ Distribution')
            axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH + 'parameter_distributions.png')


