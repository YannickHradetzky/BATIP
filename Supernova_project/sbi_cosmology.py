import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import Independent, Uniform, Normal
from sbi import inference as inference
from sbi.neural_nets.factory import posterior_nn
from sbi.analysis import pairplot
import corner
import random

# Set global random seed at the very top of the file
RANDOM_SEED = 42

# Set seeds for all random number generators
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import Independent, Uniform, Normal
from sbi import inference as inference
from sbi.neural_nets.factory import posterior_nn
from sbi.analysis import pairplot
import corner

# Set all random seeds
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.deterministic = True

# Call set_seed at the start
set_seed(RANDOM_SEED)

class SBIConfig:
    # Add random seed to config
    RANDOM_SEED = RANDOM_SEED
    
    # SBI settings
    NUM_SIMULATIONS = 10000
    NUM_POSTERIOR_SAMPLES = 100000  # Increased for better posterior visualization
    BATCH_SIZE = 128    
    # Physics constants
    C = 299792.458  # Speed of light in km/s
    
    # Data filtering
    MIN_REDSHIFT = 0.001
    
    # Paths
    DATA_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/'
    PLOT_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/'
    
    # Neural network settings
    EMBEDDING_DIM = 2
    HIDDEN_FEATURES = 64
    NUM_TRANSFORMS = 5
    
    EPSILON = 1e-5
    DEVICE = torch.device("mps")
    
    # Parameter priors (mean, std)
    PARAM_PRIORS = {
        'flat': {
            'H0': (70.0, 2.5),  # centered at 70 with std of 2.5
            'Om': (0.3, 0.05)   # centered at 0.3 with std of 0.05
        },
        'curved': {
            'H0': (70.0, 2.5),
            'Om': (0.3, 0.05),  # centered at 0.45 with std of (0.9-0)/4
            'Ok': (0.0, 0.05)      # centered at 0 with std of 0.1
        }
    }

class CosmologicalSimulator:
    def __init__(self, z_obs, model_type="flat"):
        """Initialize the simulator with observed redshift values"""
        set_seed(SBIConfig.RANDOM_SEED)  # Set seed in constructor
        self.z = torch.as_tensor(z_obs, dtype=torch.float32).clone().detach()
        self.model_type = model_type
        
    def hubble_z(self, H0, Om, Ok=None):
        """Hubble parameter at redshift z"""
        # Reshape parameters to allow broadcasting with z
        H0 = H0.reshape(-1, 1)  # Shape: (batch_size, 1)
        Om = Om.reshape(-1, 1)  # Shape: (batch_size, 1)
        
        matter_term = Om * (1 + self.z)**3
        if self.model_type == "flat":
            lambda_term = (1 - Om)
            return H0 * torch.sqrt(matter_term + lambda_term)
        else:
            Ok = Ok.reshape(-1, 1)  # Shape: (batch_size, 1)
            lambda_term = (1 - Om - Ok)
            curvature_term = Ok * (1 + self.z)**2
            return H0 * torch.sqrt(matter_term + curvature_term + lambda_term)
    
    def luminosity_distance(self, H0, Om, Ok=None):
        """Compute luminosity distance"""
        # Create a fine grid for integration starting from MIN_REDSHIFT
        z_max = torch.max(self.z)
        n_points = 1000
        z_grid = torch.linspace(SBIConfig.MIN_REDSHIFT, z_max, n_points)
        
        # Calculate H(z) on the fine grid for each parameter set
        H0_expanded = H0.reshape(-1, 1)  # Shape: (batch_size, 1)
        Om_expanded = Om.reshape(-1, 1)  # Shape: (batch_size, 1)
        
        # Calculate integrand on the fine grid
        matter_term = Om_expanded * (1 + z_grid)**3
        if self.model_type == "flat":
            lambda_term = (1 - Om_expanded)
            Hz = H0_expanded * torch.sqrt(matter_term + lambda_term)
        else:
            Ok_expanded = Ok.reshape(-1, 1)
            lambda_term = (1 - Om_expanded - Ok_expanded)
            curvature_term = Ok_expanded * (1 + z_grid)**2
            Hz = H0_expanded * torch.sqrt(matter_term + curvature_term + lambda_term)
            
        integrand = SBIConfig.C / Hz  # Shape: (batch_size, n_points)
        
        # For each redshift in self.z, integrate up to that redshift
        d_L = torch.zeros((len(H0), len(self.z)))  # Shape: (batch_size, n_redshifts)
        
        for i, z in enumerate(self.z):
            # For each redshift value, create a mask for integration
            z_val = float(z)  # Convert to float for comparison
            if z_val >= SBIConfig.MIN_REDSHIFT:  # Only integrate if z >= MIN_REDSHIFT
                mask = z_grid <= z_val
                # Integrate up to this redshift for all parameter sets at once
                chi = torch.trapz(integrand[:, mask], z_grid[mask], dim=1)  # Shape: (batch_size,)
                
                if self.model_type == "curved":
                    # Apply curvature correction
                    Ok_val = Ok.reshape(-1)
                    chi = apply_curvature_correction(chi, Ok_val)
                
                d_L[:, i] = (1 + z_val) * chi
        
        return d_L
    
    def distance_modulus(self, H0, Om, Ok=None):
        """Compute distance modulus"""
        d_L = self.luminosity_distance(H0, Om, Ok)
        return 5 * torch.log10(d_L + SBIConfig.EPSILON) + 25
    
    def simulate(self, params):
        """Simulate distance moduli for given parameters"""
        if params.ndim == 1:
            params = params.unsqueeze(0)
        
        if self.model_type == "flat":
            H0, Om = params[:, 0], params[:, 1]
            mu = self.distance_modulus(H0, Om)
        else:
            H0, Om, Ok = params[:, 0], params[:, 1], params[:, 2]
            mu = self.distance_modulus(H0, Om, Ok)
        
        # Return both z and mu for each simulation
        return torch.stack([self.z.expand(len(params), -1), mu], dim=-1)

# TODO: torch.distributions for covariance matrix 
# TODO: corner plot (corner package)
def apply_curvature_correction(chi, Ok):
    """Apply curvature correction to comoving distance, differentiating between positive and negative Ok
    
    For Ok > 0 (open universe):
        chi/sqrt(Ok) + (chi^3)/6 * sqrt(Ok) + (chi^5)/120 * sqrt(Ok)^3
    For Ok < 0 (closed universe):
        chi/sqrt(-Ok) - (chi^3)/6 * sqrt(-Ok) + (chi^5)/120 * sqrt(-Ok)^3
    """
    # Handle positive Ok
    def positive_ok(Ok, chi):
        sqrt_Ok = torch.sqrt(Ok + SBIConfig.EPSILON)
        return chi/sqrt_Ok + (chi**3)/6 * sqrt_Ok + (chi**5)/120 * sqrt_Ok**3
    
    # Handle negative Ok
    def negative_ok(Ok, chi):
        sqrt_neg_Ok = torch.sqrt(-Ok + SBIConfig.EPSILON)
        return chi/sqrt_neg_Ok - (chi**3)/6 * sqrt_neg_Ok + (chi**5)/120 * sqrt_neg_Ok**3
    
    # Use where to select appropriate calculation based on Ok sign
    result = torch.where(Ok > 0, 
                        positive_ok(Ok, chi),
                        negative_ok(Ok, chi))
    
    return result

def load_real_data():
    """Load and prepare the supernova data"""
    import pandas as pd
    df = pd.read_csv(SBIConfig.DATA_PATH + 'Pantheon+SH0ES.dat', sep='\s+')
    
    # Filter data
    mask = df['zCMB'] > SBIConfig.MIN_REDSHIFT
    z_obs = df['zCMB'][mask].values
    mu_obs = df['MU_SH0ES'][mask].values
    mu_err = df['MU_SH0ES_ERR_DIAG'][mask].values
    
    # Convert to tensors
    z_obs = torch.as_tensor(z_obs, dtype=torch.float32).clone().detach()
    mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).clone().detach()
    mu_err = torch.as_tensor(mu_err, dtype=torch.float32).clone().detach()
    
    return z_obs, mu_obs, mu_err

def plot_training_data(z_obs, mu_obs, mu_err, theta, simulated_data, model_type="flat"):
    """Create visualization of parameter space coverage and model predictions during training"""
    if model_type == "flat":
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 0.5])
        
        # Plot parameter space coverage (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(theta[:, 0], theta[:, 1], c=simulated_data.mean(dim=1), 
                            cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax1, label='Mean μ')
        ax1.set_xlabel('H₀ [km/s/Mpc]')
        ax1.set_ylabel('Ωₘ')
        ax1.set_title('Parameter Space Coverage')
        
        # Plot model predictions vs data (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.errorbar(z_obs.numpy(), mu_obs.numpy(), yerr=mu_err.numpy(),
                    fmt='r.', alpha=0.5, label='Data', markersize=2)
        
        # Plot all simulations to show distribution
        simulated_mean = simulated_data.mean(dim=0).numpy()
        simulated_std = simulated_data.std(dim=0).numpy()
        
        # Plot mean and standard deviation bands
        ax2.fill_between(z_obs.numpy(), 
                        simulated_mean - 2*simulated_std,
                        simulated_mean + 2*simulated_std,
                        color='b', alpha=0.1, label='2σ region')
        ax2.fill_between(z_obs.numpy(),
                        simulated_mean - simulated_std,
                        simulated_mean + simulated_std,
                        color='b', alpha=0.2, label='1σ region')
        ax2.plot(z_obs.numpy(), simulated_mean, 'b-', label='Mean prediction')
        
        ax2.set_xlabel('Redshift (z)')
        ax2.set_ylabel('Distance Modulus (μ)')
        ax2.set_title('Model Predictions (n=50) vs Data')
        ax2.legend()
        
        # Plot parameter distributions (bottom)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(theta[:, 0].numpy(), bins=50, density=True)
        ax3.set_xlabel('H₀ [km/s/Mpc]')
        ax3.set_ylabel('Density')
        ax3.set_title('H₀ Distribution')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(theta[:, 1].numpy(), bins=50, density=True)
        ax4.set_xlabel('Ωₘ')
        ax4.set_ylabel('Density')
        ax4.set_title('Ωₘ Distribution')
    
    else:  # curved model
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
        
        # Plot H0 vs Om parameter space (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(theta[:, 0], theta[:, 1], c=simulated_data.mean(dim=1), 
                            cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax1, label='Mean μ')
        ax1.set_xlabel('H₀ [km/s/Mpc]')
        ax1.set_ylabel('Ωₘ')
        ax1.set_title('H₀-Ωₘ Parameter Space')
        
        # Plot Om vs Ok parameter space (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(theta[:, 1], theta[:, 2], c=simulated_data.mean(dim=1), 
                            cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax2, label='Mean μ')
        ax2.set_xlabel('Ωₘ')
        ax2.set_ylabel('Ωₖ')
        ax2.set_title('Ωₘ-Ωₖ Parameter Space')
        
        # Plot model predictions vs data (middle)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.errorbar(z_obs.numpy(), mu_obs.numpy(), yerr=mu_err.numpy(),
                    fmt='r.', alpha=0.5, label='Data', markersize=2)
        
        # Plot all simulations to show distribution
        simulated_mean = simulated_data.mean(dim=0).numpy()
        simulated_std = simulated_data.std(dim=0).numpy()
        
        # Plot mean and standard deviation bands
        ax3.fill_between(z_obs.numpy(), 
                        simulated_mean - 2*simulated_std,
                        simulated_mean + 2*simulated_std,
                        color='b', alpha=0.2, label='95% CI')
        ax3.fill_between(z_obs.numpy(),
                        simulated_mean - simulated_std, 
                        simulated_mean + simulated_std,
                        color='b', alpha=0.3, label='68% CI')
        ax3.plot(z_obs.numpy(), simulated_mean, 'b-', label='Mean prediction')
        
        # Plot all individual simulations with high transparency
        for i in range(len(simulated_data)):
            ax3.plot(z_obs.numpy(), simulated_data[i].numpy(),
                    'b-', alpha=0.01)
        
        ax3.set_xlabel('Redshift (z)')
        ax3.set_ylabel('Distance Modulus (μ)')
        ax3.set_title('Model Predictions (n=50) vs Data')
        ax3.legend()
        
        # Plot parameter distributions (bottom)
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(theta[:, 0].numpy(), bins=50, density=True)
        ax4.set_xlabel('H₀ [km/s/Mpc]')
        ax4.set_ylabel('Density')
        ax4.set_title('H₀ Distribution')
        
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(theta[:, 1].numpy(), bins=50, density=True, alpha=0.5, label='Ωₘ')
        ax5.hist(theta[:, 2].numpy(), bins=50, density=True, alpha=0.5, label='Ωₖ')
        ax5.set_xlabel('Parameter Value')
        ax5.set_ylabel('Density')
        ax5.set_title('Ωₘ and Ωₖ Distributions')
        ax5.legend()
    
    plt.tight_layout()
    plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_training_data_{model_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

class CosmologySBI:
    def __init__(self, z_obs, mu_obs, mu_err):
        """Initialize the SBI trainer with observed data"""
        set_seed(SBIConfig.RANDOM_SEED)  # Set seed in constructor
        self.z_obs = z_obs
        self.mu_obs = mu_obs
        self.mu_err = mu_err
        
        # Setup simulator with default flat model
        self.simulator = CosmologicalSimulator(z_obs)
    
    def setup_prior(self):
        """Setup the prior distribution for parameters"""
        if self.simulator.model_type == "flat":
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([SBIConfig.PARAM_PRIORS['flat']['H0'][0],
                                    SBIConfig.PARAM_PRIORS['flat']['Om'][0]]),
                    scale=torch.tensor([SBIConfig.PARAM_PRIORS['flat']['H0'][1],
                                      SBIConfig.PARAM_PRIORS['flat']['Om'][1]])
                ),
                1
            )
        else:
            self.prior = Independent(
                Normal(
                    loc=torch.tensor([SBIConfig.PARAM_PRIORS['curved']['H0'][0],
                                    SBIConfig.PARAM_PRIORS['curved']['Om'][0],
                                    SBIConfig.PARAM_PRIORS['curved']['Ok'][0]]),
                    scale=torch.tensor([SBIConfig.PARAM_PRIORS['curved']['H0'][1],
                                      SBIConfig.PARAM_PRIORS['curved']['Om'][1],
                                      SBIConfig.PARAM_PRIORS['curved']['Ok'][1]])
                ),
                1
            )
    
    def create_embedding_net(self):
        """Create an embedding network for the data"""
        return nn.Sequential(
            nn.Linear(len(self.z_obs), SBIConfig.HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(SBIConfig.HIDDEN_FEATURES, SBIConfig.HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(SBIConfig.HIDDEN_FEATURES, SBIConfig.EMBEDDING_DIM),
        )
    
    def train(self):
        """Train the neural network"""
        set_seed(SBIConfig.RANDOM_SEED)  # Set seed before training
        # Setup prior based on current model type
        self.setup_prior()
        
        # Setup embedding network and neural posterior estimator
        embedding_net = self.create_embedding_net()
        
        # Initialize inference object
        self.posterior_estimator = inference.SNPE(prior=self.prior)
        
        # Generate training data
        print("\nGenerating training data...")
        theta = self.prior.sample((SBIConfig.NUM_SIMULATIONS,))
        x = self.simulator.simulate(theta)

        # Print example of sampled parameters and simulated data
        print("\nExample simulation:")
        print(f"Sampled parameters (theta[0]):")
        if self.simulator.model_type == "flat":
            print(f"H₀ = {theta[0,0]:.2f} km/s/Mpc")
            print(f"Ωₘ = {theta[0,1]:.2f}")
        else:
            print(f"H₀ = {theta[0,0]:.2f} km/s/Mpc")
            print(f"Ωₘ = {theta[0,1]:.2f}")
            print(f"Ωₖ = {theta[0,2]:.2f}")
        
        print("\nSimulated distance moduli (middle 5 redshift points):")
        for i in range(len(self.z_obs)//2 - 2, len(self.z_obs)//2 + 3):
            print(f"z = {self.z_obs[i]:.3f}: μ = {x[0,i,1]:.2f}")

        
        # Create training data visualization
        print("\nCreating training data visualization...")
        plot_training_data(self.z_obs, self.mu_obs, self.mu_err, theta, x[:, :, 1], 
                          model_type=self.simulator.model_type)
        
        # We only need the distance modulus for training
        x = x[:, :, 1]
        
        print("\nTraining the neural network...")
        # Train the network
        density_estimator = self.posterior_estimator.append_simulations(theta, x).train(
            training_batch_size=SBIConfig.BATCH_SIZE
        )
        self.posterior = self.posterior_estimator.build_posterior(density_estimator)
    
    def sample_posterior(self, num_samples=None):
        """Sample from the posterior distribution"""
        set_seed(SBIConfig.RANDOM_SEED)  # Set seed before sampling
        if num_samples is None:
            num_samples = SBIConfig.NUM_POSTERIOR_SAMPLES
        return self.posterior.sample((num_samples,), x=self.mu_obs)
    # TODO: Mit fake daten neuen Posterior bilden, für die Daten kennen wir die Parameter

    def back_test_network(self):
        """Test the network with simulated data"""
        set_seed(SBIConfig.RANDOM_SEED)  # Set seed before back testing
        # create fake data
        theta = self.prior.sample((1,))  # Sample a single set of parameters
        x = self.simulator.simulate(theta)
        # Extract just the distance modulus values (second dimension)
        x = x[0, :, 1]  # Shape should now be (n_redshifts,)
        
        # Sample from posterior using the simulated data
        samples = self.posterior.sample((SBIConfig.NUM_POSTERIOR_SAMPLES,), x=x)
        
        # Create plots
        if self.simulator.model_type == "flat":
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            omega_l = 1 - samples[:, 1]
        else:
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))
            omega_l = 1 - samples[:, 1] - samples[:, 2]
        
        # Plot distributions
        axes[0].hist(samples[:, 0], bins=50, color='skyblue', edgecolor='black', density=True)
        axes[0].set_title('H₀ Distribution')
        axes[0].set_xlabel('H₀ [km/s/Mpc]')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(samples[:, 1], bins=50, color='lightgreen', edgecolor='black', density=True)
        axes[1].set_title('Ωₘ Distribution')
        axes[1].set_xlabel('Ωₘ')
        axes[1].set_ylabel('Frequency')

        axes[2].hist(omega_l, bins=50, color='salmon', edgecolor='black', density=True)
        axes[2].set_title('Ωₗ Distribution')
        axes[2].set_xlabel('Ωₗ')
        axes[2].set_ylabel('Frequency')

        if self.simulator.model_type == "curved":
            axes[3].hist(samples[:, 2], bins=50, color='purple', edgecolor='black', density=True)
            axes[3].set_title('Ωₖ Distribution')
            axes[3].set_xlabel('Ωₖ')
            axes[3].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_back_{self.simulator.model_type}_posteriors.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_scientific_results(samples_flat=None, samples_curved=None, data_type="real"):
    """Create publication-quality plots of the results"""
    for model_type, samples in [("flat", samples_flat), ("curved", samples_curved)]:
        if samples is None:
            continue
            
        # Convert tensor to numpy array if needed
        if torch.is_tensor(samples):
            samples = samples.numpy()
            
        # Calculate Omega_lambda
        omega_l = 1 - samples[:, 1] if model_type == "flat" else 1 - samples[:, 1] - samples[:, 2]
        
        # Create figure
        if model_type == "flat":
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(f'Parameter Distributions for Flat ΛCDM Model{" (Simulated Data)" if data_type=="fake" else ""}', 
                        fontsize=16, y=1.05)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(25, 5))
            fig.suptitle(f'Parameter Distributions for Curved ΛCDM Model{" (Simulated Data)" if data_type=="fake" else ""}', 
                        fontsize=16, y=1.05)
        
        # Common plot settings
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
        
        # H0 distribution
        axes[0].hist(samples[:, 0], bins=50, color='skyblue', edgecolor='black', density=True)
        axes[0].set_title('Hubble Parameter Distribution')
        axes[0].set_xlabel('H₀ [km/s/Mpc]')
        axes[0].set_ylabel('Frequency')
        
        # Omega_m distribution
        axes[1].hist(samples[:, 1], bins=50, color='lightgreen', edgecolor='black', density=True)
        axes[1].set_title('Matter Density Distribution')
        axes[1].set_xlabel('Ωₘ')
        axes[1].set_ylabel('Frequency')
        
        # Omega_lambda distribution
        axes[2].hist(omega_l, bins=50, color='salmon', edgecolor='black', density=True)
        axes[2].set_title('Dark Energy Density Distribution')
        axes[2].set_xlabel('Ωₗ')
        axes[2].set_ylabel('Frequency')
        
        if model_type == "curved":
            # Ok distribution
            axes[3].hist(samples[:, 2], bins=50, color='purple', edgecolor='black', density=True)
            axes[3].set_title('Curvature Density Distribution')
            axes[3].set_xlabel('Ωₖ')
            axes[3].set_ylabel('Frequency')
        
        plt.tight_layout()
        if data_type == "fake":
            plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_{model_type}_posteriors_fake.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_{model_type}_posteriors.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create corner plot
        if model_type == "curved":
            fig = corner.corner(
                samples, 
                labels=['H₀', 'Ωₘ', 'Ωₖ'], 
                plot_datapoints=False, 
                plot_density=False, 
                contours=True, 
                fill_contours=True, 
                levels=[0.68, 0.95, 0.997],
                title_kwargs={"fontsize": 16},
                label_kwargs={"fontsize": 14}, 
                color="skyblue"
            )
            plt.suptitle(
                f'Corner Plot for Curved ΛCDM Model{" (Simulated Data)" if data_type=="fake" else ""}', 
                fontsize=16, 
                y=1.02
            )
        else:
            fig = corner.corner(
                samples, 
                labels=['H₀', 'Ωₘ'], 
                plot_datapoints=False, 
                plot_density=False, 
                contours=True, 
                fill_contours=True, 
                levels=[0.68, 0.95, 0.997],
                title_kwargs={"fontsize": 16},
                label_kwargs={"fontsize": 14}, 
                color="skyblue"
            )
            plt.suptitle(
                f'Corner Plot for Flat ΛCDM Model{" (Simulated Data)" if data_type=="fake" else ""}', 
                fontsize=16, 
                y=1.02
            )
            
        if data_type == "fake":
            plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_{model_type}_corner_plot_fake.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_{model_type}_corner_plot.png', dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    set_seed(RANDOM_SEED)  # Set seed at start of main
    
    # Load real data
    z_obs, mu_obs, mu_err = load_real_data()
    print(f"Using {len(z_obs)} data points after filtering")
    
    samples_dict = {}
    
    # Test both models with a few samples
    for model_type in ["curved", "flat"]:
        print(f"\nTesting {model_type.capitalize()} ΛCDM model...")
        
        # Initialize SBI
        sbi = CosmologySBI(z_obs, mu_obs, mu_err)
        sbi.simulator.model_type = model_type
        
        # Train and sample
        print(f"\nTraining {model_type} model...")
        sbi.train()
        print("\nSampling from posterior...")
        samples_dict[model_type] = sbi.sample_posterior()

        # back test the network
        sbi.back_test_network()
    
    # Create final scientific plots
    print("\nCreating final visualization...")
    plot_scientific_results(
        samples_flat=samples_dict["flat"],
        samples_curved=samples_dict["curved"],
    )