import torch 
from torch.distributions import Independent, Uniform, MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from sbi import inference
from Config import load_real_data, Config
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
        self.Om_bounds = (0.1, 0.9)
        self.Ok_bounds = (-0.5, 0.5)
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

    def bin_analysis(self, n_bins=10, num_simulations=10000, num_samples=100000, z_max=0.2):
        """
        Perform binned analysis of the supernova data.
        
        Args:
            n_bins (int): Number of bins to use
            num_simulations (int): Number of simulations for training
            num_samples (int): Number of posterior samples to generate
        """
        # Filter data for z <= z_max
        z_mask = self.z_obs <= z_max
        z_filtered = self.z_obs[z_mask]
        mu_filtered = self.mu_obs[z_mask]
        self.z_obs = z_filtered
        self.mu_obs = mu_filtered

        # Sort z values and get sorting indices
        sort_indices = np.argsort(self.z_obs)
        z_sorted = self.z_obs[sort_indices]
        mu_sorted = self.mu_obs[sort_indices]
        
        n_points = len(self.z_obs)
        target_points_per_bin = n_points // n_bins
        
        # Find gaps in the data
        z_gaps = np.diff(z_sorted)
        large_gaps = np.where(z_gaps > np.mean(z_gaps) + 2*np.std(z_gaps))[0]
        
        # Create bins considering large gaps
        z_bins = []
        mu_bins = []
        current_z_bin = []
        current_mu_bin = []
        
        for i, (z, mu) in enumerate(zip(z_sorted, mu_sorted)):
            # If we hit a large gap and have enough points, start new bin
            if i in large_gaps and len(current_z_bin) >= target_points_per_bin * 0.5:
                z_bins.append(current_z_bin)
                mu_bins.append(current_mu_bin)
                current_z_bin = []
                current_mu_bin = []
            current_z_bin.append(z)
            current_mu_bin.append(mu)
            
            # If bin is full and we're not in the last portion of data
            if len(current_z_bin) >= target_points_per_bin and i < len(z_sorted) - target_points_per_bin:
                z_bins.append(current_z_bin)
                mu_bins.append(current_mu_bin)
                current_z_bin = []
                current_mu_bin = []
        
        # Add remaining points to the last bin
        if current_z_bin:
            if len(current_z_bin) < target_points_per_bin * 0.5 and z_bins:
                # If last bin is too small, merge with previous bin
                z_bins[-1].extend(current_z_bin)
                mu_bins[-1].extend(current_mu_bin)
            else:
                z_bins.append(current_z_bin)
                mu_bins.append(current_mu_bin)
        
        # Print results
        for i, (z_bin, mu_bin) in enumerate(zip(z_bins, mu_bins)):
            z_min = min(z_bin)
            z_max = max(z_bin)
            print(f"Bin {i}: {len(z_bin)} points, z range: [{z_min:.3f}, {z_max:.3f}]")
        
        # Print statistics about the binning
        bin_sizes = [len(b) for b in z_bins]
        print(f"\nBin statistics:")
        print(f"Mean bin size: {np.mean(bin_sizes):.1f}")
        print(f"Min bin size: {np.min(bin_sizes)}")
        print(f"Max bin size: {np.max(bin_sizes)}")

        # write the z_ranges to a file
        with open(Config.DATA_PATH + f'z_ranges.txt', 'w') as f:
            for z_bin in z_bins:
                f.write(f"{min(z_bin):.3f} - {max(z_bin):.3f}\n")

        # create a covariance matrix for each bin
        cov_matrices = []
        for mu_bin in mu_bins:
            mu_array = np.array(mu_bin)
            cov_matrix = np.diag(np.ones(len(mu_array)) * np.var(mu_array))
            cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32)
            cov_matrices.append(cov_matrix)

        # Store original data
        original_z = self.z_obs
        original_mu = self.mu_obs
        original_cov = self.cov_matrix
        
        # perform the sbi inference for each bin
        samples = []
        try:
            for i, (z_bin, mu_bin, cov_matrix) in enumerate(zip(z_bins, mu_bins, cov_matrices)):
                print(f"Performing sbi inference for bin {i}")
                # Update model data for this bin
                self.z_obs = np.array(z_bin)
                self.mu_obs = torch.tensor(mu_bin, dtype=torch.float32)
                self.cov_matrix = cov_matrix
                
                # Train and sample
                self.num_simulations = num_simulations
                self.train()
                bin_samples = self.sample_posterior(num_samples)
                samples.append(bin_samples)
                
                # Save samples
                np.savetxt(Config.DATA_PATH + f'sbi_samples_{i}_{self.type}.txt', bin_samples)
        
        finally:
            # Restore original data
            self.z_obs = original_z
            self.mu_obs = original_mu
            self.cov_matrix = original_cov
        
        return samples, z_bins, mu_bins


if __name__ == "__main__":
    model = sbi_model(type="curved")
    model.bin_analysis(
        n_bins=15,
        num_simulations=10000,
        num_samples=100000,
        z_max=0.2
    )
    

    # dict_storage = {}
    # n_samples = 10000
    # n_simulations = 1000
    # for type in ["curved", "flat"]:
    #     print(f"Training {type} model")
    #     model = sbi_model(num_simulations=n_simulations, type=type)
    #     model.train()
    #     # Now sample with the actual observed data
    #     print(f"Sampling from {type} model")
    #     samples = model.sample_posterior(n_samples)
    #     # calculate Ol
    #     if type == "curved":
    #         Ol = 1 - samples[:, 1] - samples[:, 2]
    #     else:
    #         Ol = 1 - samples[:, 1]
    #     samples = np.column_stack((samples, Ol))
    #     dict_storage[type] = samples

    #     # print the mean and std of the samples
    #     print(f"\nResults for {type} model:")
    #     print("Parameter      Mean ± Std")
    #     print("-" * 30)
    #     print(f"H0:         {samples[:, 0].mean():.2f} ± {samples[:, 0].std():.2f}")
    #     print(f"Ωm:         {samples[:, 1].mean():.3f} ± {samples[:, 1].std():.3f}")
    #     if type == "curved":
    #         print(f"Ωk:         {samples[:, 2].mean():.3f} ± {samples[:, 2].std():.3f}")
    #     print(f"ΩΛ:         {samples[:, -1].mean():.3f} ± {samples[:, -1].std():.3f}")
    #     print()
        
    #     # write samples to file
    #     print(f"Writing samples to file for {type} model")
    #     np.savetxt(Config.DATA_PATH + f'sbi_samples_{type}_{n_simulations}_{n_samples}.txt', samples)

            
        
        
    #     # create histograms in subplots
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
    #     ax1.hist(Ho, bins=50, color='blue', edgecolor='black', density=True)
    #     ax1.set_title('H0 Distribution')
        
    #     ax2.hist(Om, bins=50, color='green', edgecolor='black', density=True)
    #     ax2.set_title('Ωm Distribution')
        
    #     if type == "curved":
    #         ax3.hist(Ok, bins=50, color='red', edgecolor='black', density=True)
    #         ax3.set_title('Ωk Distribution')
    #     else:
    #         ax3.set_visible(False)
        
    #     ax4.hist(Ol, bins=50, color='purple', edgecolor='black', density=True)
    #     ax4.set_title('ΩΛ Distribution')
        
    #     plt.tight_layout()
    #     plt.savefig(PLOT_PATH + f'parameter_distributions_{type}.png')

    # # plot the distributions for each type
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # colors = {"flat": "skyblue", "curved": "orange"}
    # alpha = 0.7
    
    # for type in ["flat", "curved"]:
    #     axs[0, 0].hist(dict_storage[type][:, 0], bins=50, color=colors[type], 
    #                    edgecolor='black', density=True, alpha=alpha, label=type)
    #     axs[0, 0].set_title('H0 Distribution')
    #     axs[0, 0].legend()
        
    #     axs[0, 1].hist(dict_storage[type][:, 1], bins=50, color=colors[type], 
    #                    edgecolor='black', density=True, alpha=alpha, label=type)
    #     axs[0, 1].set_title('Ωm Distribution')
    #     axs[0, 1].legend()

    #     if type == "curved":
    #         axs[1, 0].hist(dict_storage[type][:, 2], bins=50, color=colors[type], 
    #                    edgecolor='black', density=True, alpha=alpha, label=type)
    #         axs[1, 0].set_title('Ωk Distribution')
    #         axs[1, 0].legend()
    #     else:
    #         axs[1, 0].set_visible(False)
        
    #     if type == "curved":
    #         axs[1, 1].hist(1 - dict_storage[type][:, 1] - dict_storage[type][:, 2], 
    #                    bins=50, color=colors[type], edgecolor='black', 
    #                    density=True, alpha=alpha, label=type)
    #         axs[1, 1].set_title('ΩΛ Distribution')
    #         axs[1, 1].legend()
    #     else:
    #         axs[1, 1].hist(1 - dict_storage[type][:, 1], bins=50, color=colors[type], 
    #                    edgecolor='black', density=True, alpha=alpha, label=type)
    #         axs[1, 1].set_title('ΩΛ Distribution')
    #         axs[1, 1].legend()
    
    # plt.tight_layout()
    # plt.savefig(PLOT_PATH + 'parameter_distributions.png')


