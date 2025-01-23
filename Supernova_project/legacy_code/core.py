import torch
import numpy as np
import matplotlib.pyplot as plt
import corner
from sbi.utils import plot_predictive
from sbi.analysis import pairplot
from Supernova_project.legacy_code.sbi_cosmology import SBIConfig


def plot_model_comparison(samples_flat, samples_curved, data_type="real"):
    """Create comparison plots between flat and curved models"""
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(samples_flat):
        samples_flat = samples_flat.numpy()
    if torch.is_tensor(samples_curved):
        samples_curved = samples_curved.numpy()
    
    # Calculate Omega_lambda for both models
    omega_l_flat = 1 - samples_flat[:, 1]
    omega_l_curved = 1 - samples_curved[:, 1] - samples_curved[:, 2]
    
    # Create figure for parameter distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Adjust the spacing between plots and title
    plt.subplots_adjust(top=0.85)  # Increase space for title
    
    fig.suptitle('Comparison of Flat and Curved ΛCDM Models' + 
                (" (Simulated Data)" if data_type=="fake" else ""),
                fontsize=16, y=0.98)  # Move title higher
    
    ranges = [
        (65, 75),      # H₀ range
        (0.1, 0.5),    # Ωₘ range
        (-0.2, 0.2)    # Ωₖ range
    ]
    
    # Common plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # H0 distribution comparison
    axes[0,0].hist(samples_flat[:, 0], bins=50, alpha=0.5, color='skyblue', 
                   label='Flat', density=True)
    axes[0,0].hist(samples_curved[:, 0], bins=50, alpha=0.5, color='salmon', 
                   label='Curved', density=True)
    axes[0,0].set_title('Hubble Parameter Distribution')
    axes[0,0].set_xlabel('H₀ [km/s/Mpc]')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    
    # Omega_m distribution comparison
    axes[0,1].hist(samples_flat[:, 1], bins=50, alpha=0.5, color='skyblue', 
                   label='Flat', density=True)
    axes[0,1].hist(samples_curved[:, 1], bins=50, alpha=0.5, color='salmon', 
                   label='Curved', density=True)
    axes[0,1].set_title('Matter Density Distribution')
    axes[0,1].set_xlabel('Ωₘ')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    
    # Omega_lambda distribution comparison
    axes[1,0].hist(omega_l_flat, bins=50, alpha=0.5, color='skyblue', 
                   label='Flat', density=True)
    axes[1,0].hist(omega_l_curved, bins=50, alpha=0.5, color='salmon', 
                   label='Curved', density=True)
    axes[1,0].set_title('Dark Energy Density Distribution')
    axes[1,0].set_xlabel('Ωₗ')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    
    # Omega_k distribution (curved model only)
    axes[1,1].hist(samples_curved[:, 2], bins=50, color='salmon', 
                   label='Curved', density=True)
    axes[1,1].set_title('Curvature Density Distribution')
    axes[1,1].set_xlabel('Ωₖ')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    
    plt.tight_layout()
    if data_type == "fake":
        plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_model_comparison_posteriors_fake.png', 
                   dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{SBIConfig.PLOT_PATH}sbi_model_comparison_posteriors.png', 
                   dpi=300, bbox_inches='tight')
    plt.close()


def plot_autocorrelation(samples, model_type):
    """Calculate the autocorrelation length for each parameter"""
    # Convert to numpy array if needed
    if torch.is_tensor(samples):
        samples = samples.numpy()
    
    # Number of parameters
    n_params = samples.shape[1]
    
    # Create figure
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    # Parameter names
    param_names = ['H₀', 'Ωₘ'] if model_type == "flat" else ['H₀', 'Ωₘ', 'Ωₖ']
    
    corr_lengths = []
    # Calculate autocorrelation for each parameter
    for i in range(n_params):
        # Calculate autocorrelation length and function
        corr_length, autocorr = autocorrelation_length(samples[:, i])
        corr_lengths.append(corr_length)
        
        # Create lag array for plotting
        lags = np.arange(len(autocorr))
        
        # Plot
        axes[i].plot(lags, autocorr)
        axes[i].axhline(y=np.exp(-1), color='r', linestyle='--', alpha=0.5, 
                       label='e⁻¹ threshold')
        if corr_length is not None:
            axes[i].axvline(x=corr_length, color='g', linestyle='--', alpha=0.5,
                          label=f'Correlation length: {corr_length}')
        
        axes[i].set_title(f'{param_names[i]}')
        axes[i].set_xlabel('Lag')
        axes[i].set_ylabel('Autocorrelation')
        axes[i].legend()
        
        print(f"Correlation length for {param_names[i]}: {corr_length}")
    
    plt.tight_layout()
    plt.savefig(f'{SBIConfig.PLOT_PATH}autocorrelation_{model_type}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the maximum correlation length
    max_corr_length = max(length for length in corr_lengths if length is not None)
    return max_corr_length

def autocorrelation_length(data):
    """
    Calculate the autocorrelation length of a given array.
    
    Parameters:
    - data (array-like): Input array of numerical values.

    Returns:
    - corr_length (float): The correlation length where autocorrelation falls to e^-1.
    - autocorr (np.ndarray): Autocorrelation values for each lag.
    """
    # Ensure input is a numpy array
    data = np.asarray(data)
    n = len(data)
    
    # Subtract the mean
    data_mean = np.mean(data)
    data -= data_mean
    
    # Compute the autocorrelation function
    autocorr = np.correlate(data, data, mode='full')[n-1:] / (np.var(data) * n)
    
    # Find the correlation length (first lag where autocorr <= e^-1)
    threshold = np.exp(-1)
    corr_length = next((lag for lag, value in enumerate(autocorr) if value <= threshold), None)
    
    return corr_length, autocorr

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
