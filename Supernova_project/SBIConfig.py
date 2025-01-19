import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
class SBIConfig:
    # Add random seed to config
    RANDOM_SEED = 42
    
    # SBI settings
    NUM_SIMULATIONS = 10000
    NUM_POSTERIOR_SAMPLES = 1000000  # Increased for better posterior visualization
    BATCH_SIZE = 256
    # Physics constants
    C = 299792.458  # Speed of light in km/s
    
    # Data filtering
    MIN_REDSHIFT = 0.001
    
    # Paths
    DATA_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/'
    PLOT_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/'
    # Load real data
    
    # General settings
    EPSILON = 1e-5
    DEVICE = torch.device("cpu")
    
    # Parameter priors (mean, std)
    PARAM_PRIORS = {
        'flat': {
            'H0': (70.0, 1.5),  # Tighter constraint around 71
            'Om': (0.2, 0.03)   # Tighter constraint around 0.3
        },
        'curved': {
            'H0': (70.0, 1.5),
            'Om': (0.2, 0.03),
            'Ok': (0.0, 0.01)  # Much tighter constraint around 0
        }
    }


def load_real_data():
    """Load and prepare the supernova data"""
    import pandas as pd
    df = pd.read_csv('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES.dat', sep='\s+')
    
    # Filter data
    mask = df['zCMB'] > 0.001
    z_obs = df['zCMB'][mask].values
    mu_obs = df['MU_SH0ES'][mask].values
    mu_err = df['MU_SH0ES_ERR_DIAG'][mask].values
    
    # Convert to tensors
    z_obs = torch.as_tensor(z_obs, dtype=torch.float32).clone().detach()
    mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).clone().detach()
    mu_err = torch.as_tensor(mu_err, dtype=torch.float32).clone().detach()

    # Create covariance matrix
    cov_data = np.loadtxt('/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Data/Pantheon+SH0ES_STATONLY.cov')
    cov_matrix = _create_cov_mat(cov_data)
    
    return z_obs, mu_obs, mu_err, cov_matrix

def _create_cov_mat(cov_data):
    n = int(cov_data[0])
    cov_matrix = cov_data[1:].reshape(n, n)
    
    # Make symmetric if needed
    if not np.allclose(cov_matrix, cov_matrix.T):
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
    return cov_matrix

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

# Load real data
z_obs, mu_obs, mu_err, cov_matrix = load_real_data()