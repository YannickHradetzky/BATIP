import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Independent, Uniform

from Supernova_project.legacy_code.sbi_cosmology import CosmologySBI, SBIConfig

def test_simulator(min_redshift=0.01):
    # Initialize the SBI setup
    sbi = CosmologySBI()
    
    # Filter out very low redshifts
    mask = sbi.z >= min_redshift
    sbi.z = sbi.z[mask]
    sbi.mu_obs = sbi.mu_obs[mask]
    sbi.mu_err = sbi.mu_err[mask]
    
    print(f"Using {len(sbi.z)} data points after filtering z >= {min_redshift}")
    
    # Setup prior for flat ΛCDM model
    prior = Independent(
        Uniform(
            low=torch.tensor([SBIConfig.PARAM_BOUNDS['flat']['H0'][0],
                            SBIConfig.PARAM_BOUNDS['flat']['Om'][0]], 
                           device=sbi.device),
            high=torch.tensor([SBIConfig.PARAM_BOUNDS['flat']['H0'][1],
                             SBIConfig.PARAM_BOUNDS['flat']['Om'][1]], 
                            device=sbi.device)
        ),
        1
    )
    
    # Generate test parameters
    n_test = 10000
    test_theta = prior.sample((n_test,))
    
    # Run simulator
    print("Running simulator with test parameters...")
    test_x = torch.stack([sbi.simulator_flat(theta_i) for theta_i in test_theta])
    
    # Print parameter statistics
    print("\nParameter Statistics:")
    print(f"H₀ range: [{test_theta[:, 0].min():.2f}, {test_theta[:, 0].max():.2f}]")
    print(f"H₀ mean ± std: {test_theta[:, 0].mean():.2f} ± {test_theta[:, 0].std():.2f}")
    print(f"Ωₘ range: [{test_theta[:, 1].min():.2f}, {test_theta[:, 1].max():.2f}]")
    print(f"Ωₘ mean ± std: {test_theta[:, 1].mean():.2f} ± {test_theta[:, 1].std():.2f}")
    
    # Print output statistics
    print("\nSimulator Output Statistics:")
    print(f"μ range: [{test_x.min():.2f}, {test_x.max():.2f}]")
    print(f"μ mean ± std: {test_x.mean():.2f} ± {test_x.std():.2f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot parameter space
    scatter = axes[0,0].scatter(test_theta[:, 0].numpy(), 
                               test_theta[:, 1].numpy(),
                               c=test_x.mean(dim=1).numpy(),
                               cmap='viridis')
    axes[0,0].set_xlabel('H₀')
    axes[0,0].set_ylabel('Ωₘ')
    axes[0,0].set_title('Parameter Space Coverage')
    plt.colorbar(scatter, ax=axes[0,0], label='Mean μ')
    
    # Plot simulator outputs vs data
    for i in range(min(20, len(test_x))):
        axes[0,1].plot(sbi.z.numpy(), test_x[i].numpy(), 
                      alpha=0.3, color='blue', label='Model' if i == 0 else None)
    axes[0,1].scatter(sbi.z.numpy(), sbi.mu_obs.numpy(), 
                     alpha=0.5, c='red', s=10, label='Data')
    axes[0,1].set_xlabel('Redshift')
    axes[0,1].set_ylabel('Distance Modulus')
    axes[0,1].set_title('Simulator Outputs vs Data')
    axes[0,1].legend()
    
    # Plot residuals
    residuals = test_x - sbi.mu_obs
    mean_residual = residuals.mean(dim=0)
    std_residual = residuals.std(dim=0)
    
    axes[1,0].fill_between(sbi.z.numpy(), 
                          (mean_residual - std_residual).numpy(),
                          (mean_residual + std_residual).numpy(),
                          alpha=0.3, color='blue')
    axes[1,0].plot(sbi.z.numpy(), mean_residual.numpy(), 'b-', label='Mean Residual')
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Redshift')
    axes[1,0].set_ylabel('Residual (Model - Data)')
    axes[1,0].set_title('Residuals Analysis')
    axes[1,0].legend()
    
    # Plot residual distribution
    axes[1,1].hist(residuals.numpy().flatten(), bins=50, density=True)
    axes[1,1].set_xlabel('Residual Value')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{SBIConfig.PLOT_PATH}simulator_test.png')
    plt.close()
    
    # Check for numerical issues
    print("\nNumerical Checks:")
    print(f"NaN in outputs: {torch.isnan(test_x).any()}")
    print(f"Inf in outputs: {torch.isinf(test_x).any()}")
    
    return test_theta, test_x

if __name__ == "__main__":
    test_theta, test_x = test_simulator(min_redshift=0.01)