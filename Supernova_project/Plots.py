import matplotlib.pyplot as plt
import numpy as np

PLOT_PATH = '/Users/yhra/Documents/Master/Semester_3/BATIP/Supernova_project/Plots/'

def plot_training_data(theta, z_obs, mu_obs, simulated_data, type):
    """Plot the training data and parameter space coverage"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # plot parameter space coverage
    ax1.scatter(theta[:, 0], theta[:, 1])
    ax1.set_xlabel('H0')
    ax1.set_ylabel('Ωm')
    ax1.set_title('Parameter Space Coverage')

    # plot parameter space coverage
    ax2.scatter(theta[:, 1], theta[:, 2])
    ax2.set_xlabel('Ωm')
    ax2.set_ylabel('Ωk')
    ax2.set_title('Parameter Space Coverage')

    # plot distance modulus predictions
    # Make sure z_obs matches the dimension of simulated_data
    z_obs_expanded = z_obs.unsqueeze(0).expand(simulated_data.shape[0], -1)
    
    ax3.plot(z_obs, simulated_data.mean(dim=0), 'b-', label='Mean prediction')
    ax3.fill_between(z_obs, 
                     simulated_data.mean(dim=0) - simulated_data.std(dim=0), 
                     simulated_data.mean(dim=0) + simulated_data.std(dim=0), 
                     alpha=0.2)
    ax3.fill_between(z_obs, 
                     simulated_data.mean(dim=0) - 2*simulated_data.std(dim=0), 
                     simulated_data.mean(dim=0) + 2*simulated_data.std(dim=0), 
                     alpha=0.1)
    ax3.scatter(z_obs, mu_obs, c='red', label='Observed data', zorder=3)
    ax3.set_xlabel('Redshift (z)')
    ax3.set_ylabel('Distance Modulus (μ)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH + 'training_data.png')
    