import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
class Config:
    # Add random seed to config
    RANDOM_SEED = 42
    
    # SBI settings
    NUM_SIMULATIONS = 10000
    NUM_POSTERIOR_SAMPLES = 100000  # Increased for better posterior visualization
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
            'H0': (70.0, 1.5),  # Tighter constraint around 70
            'Om': (0.3, 0.05)   # Tighter constraint around 0.3
        },
        'curved': {
            'H0': (70.0, 1.5),
            'Om': (0.3, 0.05),
            'Ok': (0.0, 0.000001)  # Much tighter constraint around 0
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
    def create_parameter_scatter(ax, x, y, data, xlabel, ylabel, title):
        scatter = ax.scatter(x, y, c=data, cmap='viridis', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax

    def plot_predictions(ax, z_obs, mu_obs, mu_err, simulated_mean, simulated_std, include_bands=False):
        ax.errorbar(z_obs.numpy(), mu_obs.numpy(), yerr=mu_err.numpy(),
                   fmt='r.', alpha=0.5, label='Data', markersize=2)
        
        if include_bands:
            ax.plot(z_obs.numpy(), simulated_mean, 'b-', label='Mean prediction')
            ax.fill_between(z_obs.numpy(), simulated_mean - simulated_std, simulated_mean + simulated_std, alpha=0.2)
            ax.fill_between(z_obs.numpy(), simulated_mean - 2*simulated_std, simulated_mean + 2*simulated_std, alpha=0.1)
        else:
            # plot all simulated predictions
            for i in range(len(simulated_data)):
                ax.plot(z_obs.numpy(), simulated_data[i].numpy(), 
                       'b-', alpha=0.01, zorder=1)
        ax.set_xlabel('Redshift (z)')
        ax.set_ylabel('Distance Modulus (μ)')
        ax.legend()
        return ax

    def create_hist(ax, data, xlabel, ylabel='Density', title=None):
        if isinstance(data, tuple):  # For multiple histograms
            for d, label in data:
                ax.hist(d, bins=50, density=True, alpha=0.5, label=label)
            ax.legend()
        else:
            ax.hist(data, bins=50, density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        return ax

    # Calculate common values
    simulated_mean = simulated_data.mean(dim=0).numpy()
    simulated_std = simulated_data.std(dim=0).numpy()

    if model_type == "flat":
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 0.5])
        
        # Parameter space and predictions
        create_parameter_scatter(fig.add_subplot(gs[0, 0]), 
                               theta[:, 0], theta[:, 1], simulated_data.mean(dim=1),
                               'H₀ [km/s/Mpc]', 'Ωₘ', 'Parameter Space Coverage')
        
        plot_predictions(fig.add_subplot(gs[0, 1]), 
                        z_obs, mu_obs, mu_err, simulated_mean, simulated_std, include_bands=True)
        
        # Parameter distributions
        create_hist(fig.add_subplot(gs[1, 0]), 
                   theta[:, 0].numpy(), 'H₀ [km/s/Mpc]', title='H₀ Distribution')
        create_hist(fig.add_subplot(gs[1, 1]), 
                   theta[:, 1].numpy(), 'Ωₘ', title='Ωₘ Distribution')
    
    else:  # curved model
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 0.5])
        
        # Parameter spaces
        create_parameter_scatter(fig.add_subplot(gs[0, 0]), 
                               theta[:, 0], theta[:, 1], simulated_data.mean(dim=1),
                               'H₀ [km/s/Mpc]', 'Ωₘ', 'H₀-Ωₘ Parameter Space')
        
        create_parameter_scatter(fig.add_subplot(gs[0, 1]), 
                               theta[:, 1], theta[:, 2], simulated_data.mean(dim=1),
                               'Ωₘ', 'Ωₖ', 'Ωₘ-Ωₖ Parameter Space')
        
        # Predictions
        plot_predictions(fig.add_subplot(gs[1, :]), 
                        z_obs, mu_obs, mu_err, simulated_mean, simulated_std,
                        include_bands=True).set_title('Mean Model Predictions vs Data')
        
        # Parameter distributions
        create_hist(fig.add_subplot(gs[2, 0]), 
                   theta[:, 0].numpy(), 'H₀ [km/s/Mpc]', title='H₀ Distribution')
        create_hist(fig.add_subplot(gs[2, 1]), 
                   ((theta[:, 1].numpy(), 'Ωₘ'), (theta[:, 2].numpy(), 'Ωₖ')),
                   'Parameter Value', title='Ωₘ and Ωₖ Distributions')
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOT_PATH}sbi_training_data_{model_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_scientific_results(samples_flat=None, samples_curved=None):
    """Create publication-quality plots of the results"""
    # Common plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    for model_type, samples in [("flat", samples_flat), ("curved", samples_curved)]:
        if samples is None:
            continue
            
        # Convert tensor to numpy array if needed
        if torch.is_tensor(samples):
            samples = samples.numpy()
            
        # Calculate Omega_lambda
        omega_l = 1 - samples[:, 1] if model_type == "flat" else 1 - samples[:, 1] - samples[:, 2]
        
        # Setup plot configurations
        plot_configs = [
            {
                'data': samples[:, 0],
                'color': 'skyblue',
                'title': 'Hubble Parameter Distribution',
                'xlabel': 'H₀ [km/s/Mpc]'
            },
            {
                'data': samples[:, 1],
                'color': 'lightgreen',
                'title': 'Matter Density Distribution',
                'xlabel': 'Ωₘ'
            },
            {
                'data': omega_l,
                'color': 'salmon',
                'title': 'Dark Energy Density Distribution',
                'xlabel': 'Ωₗ'
            }
        ]
        
        if model_type == "curved":
            plot_configs.append({
                'data': samples[:, 2],
                'color': 'purple',
                'title': 'Curvature Density Distribution',
                'xlabel': 'Ωₖ'
            })
        
        # Create figure
        fig, axes = plt.subplots(1, len(plot_configs), 
                                figsize=(20 if model_type == "flat" else 25, 5))
        fig.suptitle(f'Parameter Distributions for {model_type.capitalize()} ΛCDM Model', 
                    fontsize=16, y=1.05)
        
        # Create distribution plots
        for ax, config in zip(axes, plot_configs):
            ax.hist(config['data'], bins=50, color=config['color'], 
                   edgecolor='black', density=True)
            ax.set_title(config['title'])
            ax.set_xlabel(config['xlabel'])
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{Config.PLOT_PATH}sbi_{model_type}_posteriors.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create corner plot
        labels = ['H₀', 'Ωₘ'] if model_type == "flat" else ['H₀', 'Ωₘ', 'Ωₖ']
        fig = corner.corner(
            samples, 
            labels=labels,
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
            f'Corner Plot for {model_type.capitalize()} ΛCDM Model', 
            fontsize=16, 
            y=1.02
        )
        plt.savefig(f'{Config.PLOT_PATH}sbi_{model_type}_corner_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_model_comparison(samples_flat, samples_curved):
    """Create comparison plots between flat and curved models"""
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(samples_flat):
        samples_flat = samples_flat.numpy()
    if torch.is_tensor(samples_curved):
        samples_curved = samples_curved.numpy()
    
    # Calculate Omega_lambda for both models
    omega_l_flat = 1 - samples_flat[:, 1]
    omega_l_curved = 1 - samples_curved[:, 1] - samples_curved[:, 2]
    
    # Create figure and adjust layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(top=0.85)
    fig.suptitle('Comparison of Flat and Curved ΛCDM Models', fontsize=16, y=0.98)
    
    # Common plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Plot distributions
    plot_configs = [
        {
            'ax': axes[0,0],
            'data': [(samples_flat[:, 0], 'Flat'), (samples_curved[:, 0], 'Curved')],
            'title': 'Hubble Parameter Distribution',
            'xlabel': 'H₀ [km/s/Mpc]'
        },
        {
            'ax': axes[0,1],
            'data': [(samples_flat[:, 1], 'Flat'), (samples_curved[:, 1], 'Curved')],
            'title': 'Matter Density Distribution',
            'xlabel': 'Ωₘ'
        },
        {
            'ax': axes[1,0],
            'data': [(omega_l_flat, 'Flat'), (omega_l_curved, 'Curved')],
            'title': 'Dark Energy Density Distribution',
            'xlabel': 'Ωₗ'
        },
        {
            'ax': axes[1,1],
            'data': [(samples_curved[:, 2], 'Curved')],
            'title': 'Curvature Density Distribution',
            'xlabel': 'Ωₖ'
        }
    ]
    
    colors = {'Flat': 'skyblue', 'Curved': 'salmon'}
    
    for config in plot_configs:
        ax = config['ax']
        for data, label in config['data']:
            ax.hist(data, bins=50, alpha=0.5, color=colors[label], 
                   label=label, density=True)
        ax.set_title(config['title'])
        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOT_PATH}sbi_model_comparison_posteriors.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Load real data
z_obs, mu_obs, mu_err, cov_matrix = load_real_data()