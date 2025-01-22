import numpyro
import jax.numpy as jnp
from Config import load_real_data
from Plots import PLOT_PATH
from numpyro.infer import MCMC, NUTS
from numpyro import sample
from numpyro.distributions import MultivariateNormal, Independent, Normal
from jax import random
import jax
import matplotlib.pyplot as plt
import time

jax.config.update("jax_platform_name", "cpu")  # Add this at the top of the file
# tell jax to use more cores
NUM_CHAINS = 1
numpyro.set_host_device_count(NUM_CHAINS)

class physics_model_single:
    def __init__(self):
        self.C = 299792.458 # km/s
        self.eps = 1e-10

    def calc_hubble_distance(self, hubble, matter, curv, z):
        matter_term = matter * (1 + z)**3
        curvature_term = curv * (1 + z)**2
        lambda_term = 1 - matter - curv
        return hubble * jnp.sqrt(matter_term + curvature_term + lambda_term)
    
    def calc_comoving_distance(self, hubble, matter, curv, z):
        z_values = jnp.linspace(0, z, 100)
        hubble_distances = self.calc_hubble_distance(hubble, matter, curv, z_values)
        dx = z_values[1] - z_values[0]
        integrand = 1 / hubble_distances
        integral = dx * (0.5 * integrand[0] + jnp.sum(integrand[1:-1]) + 0.5 * integrand[-1])
        return self.C * integral
    
    def calc_luminosity_distance(self, comoving_distance, Ok, z):
        # Handle all cases using jnp.where instead of if/else
        Ok_abs = jnp.abs(Ok)
        sqrt_Ok = jnp.sqrt(Ok_abs)
        
        # Flat universe case
        flat_case = (1 + z) * comoving_distance
        
        # Open universe case (Ok > 0)
        sinh_term = jnp.sinh(sqrt_Ok * comoving_distance / self.C)
        open_case = (1 + z) * self.C / sqrt_Ok * sinh_term
        
        # Closed universe case (Ok < 0)
        sin_term = jnp.sin(sqrt_Ok * comoving_distance / self.C)
        closed_case = (1 + z) * self.C / sqrt_Ok * sin_term
        
        # Combine all cases
        return jnp.where(
            Ok_abs < self.eps,
            flat_case,  # if flat
            jnp.where(Ok > 0, open_case, closed_case)  # if curved
        )
    
    def calc_apparent_magnitude(self, luminosity_distance):
        return 5 * jnp.log10(luminosity_distance) + 25

class physics_model:
    def __init__(self):
        self.calc = physics_model_single()
        self.test()

    def calc_hubble_distance(self, hubble, matter, curv, z):
        return self.calc.calc_hubble_distance(hubble, matter, curv, z)
    
    def calc_comoving_distance(self, hubble, matter, curv, z):
        return self.calc.calc_comoving_distance(hubble, matter, curv, z)
    
    def calc_luminosity_distance(self, comoving_distance, Ok, z):
        return self.calc.calc_luminosity_distance(comoving_distance, Ok, z)
    
    def calc_apparent_magnitude(self, luminosity_distance):
        return self.calc.calc_apparent_magnitude(luminosity_distance)
    
    def test(self):
        # test for fixed parameters
        hubble = jnp.array(70)
        matter = jnp.array(0.3)
        curv = jnp.array(0.1)
        z = jnp.array(0.1)
        comoving_distance = self.calc_comoving_distance(hubble, matter, curv, z)
        luminosity_distance = self.calc_luminosity_distance(comoving_distance, curv, z)
        # real values 
        real = {
            "comoving_distance": 416.5,
            "luminosity_distance": 458.2,
        }
        # round results
        comoving_distance = float(comoving_distance)
        luminosity_distance = float(luminosity_distance)
        print(f"comoving_distance: {comoving_distance}, luminosity_distance: {luminosity_distance}")
        # check if results are close to real values
        if abs(comoving_distance - real["comoving_distance"]) < 0.1 and \
           abs(luminosity_distance - real["luminosity_distance"]) < 0.1:
            print("Test passed")
        else:
            print("Test failed")

class Numyro_inference:
    def __init__(self, physics_model):
        self.physics_model = physics_model
        self.Ho_bounds = (60, 80)
        self.Om_bounds = (0.1, 0.5)
        self.Ok_bounds = (-0.2, 0.2)
        self.z_obs, self.mu_obs, self.mu_err, self.cov_matrix = load_real_data()
        # turn them into jax arrays
        self.z_obs = jnp.array(self.z_obs)
        self.mu_obs = jnp.array(self.mu_obs)
        self.mu_err = jnp.array(self.mu_err)
        self.cov_matrix = jnp.array(self.cov_matrix)

    def model(self, *args, **kwargs):
        # Sample from priors with broader bounds
        Ho = numpyro.sample("Ho", numpyro.distributions.Uniform(50, 90))
        Om = numpyro.sample("Om", numpyro.distributions.Uniform(0.0, 0.6))
        Ok = numpyro.sample("Ok", numpyro.distributions.Uniform(-0.5, 0.5))

        # Calculate model predictions for all z values at once
        comoving_distance = self.physics_model.calc_comoving_distance(Ho, Om, Ok, self.z_obs)
        luminosity_distance = self.physics_model.calc_luminosity_distance(comoving_distance, Ok, self.z_obs)
        mu_model = self.physics_model.calc_apparent_magnitude(luminosity_distance)
        
        # Add a small diagonal term to ensure covariance matrix is well-conditioned
        jitter = 1e-6 * jnp.eye(len(self.z_obs))
        cov = self.cov_matrix + jitter
        
        # Sample from the likelihood
        numpyro.sample(
            "obs",
            numpyro.distributions.MultivariateNormal(mu_model, cov),
            obs=self.mu_obs
        )

    def run_inference(self, num_warmup=5000, num_samples=5000):
        # Initialize the NUTS kernel with larger step size
        init_strategy = numpyro.infer.init_to_uniform()
        
        kernel = NUTS(
            self.model,
            step_size=1.0,  # Larger step size
            adapt_step_size=True,
            init_strategy=init_strategy,
            max_tree_depth=10  # Allow for more exploration
        )
        
        mcmc = MCMC(
            kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            num_chains=NUM_CHAINS,
            chain_method='parallel',
            progress_bar=True
        )
        
        # Use different random key for each run
        rng_key = random.PRNGKey(int(time.time()))
        mcmc.run(rng_key)
        return mcmc
    

if __name__ == "__main__":
    inference = Numyro_inference(physics_model())
    mcmc = inference.run_inference(num_warmup=1000, num_samples=1000)
    samples = mcmc.get_samples()
    mcmc.print_summary()
    Om = samples["Om"]
    Ok = samples["Ok"]
    Ho = samples["Ho"]
    Ol = 1 - Om - Ok
    print(Om)
    print(Ok)
    print(Ho)
    print(Ol)



    # z_obs, mu_obs, mu_err, cov_matrix = load_real_data()
    # inference.z_obs = jnp.array(z_obs)
    # inference.mu_obs = jnp.array(mu_obs)
    # inference.mu_err = jnp.array(mu_err)
    # inference.cov_matrix = jnp.array(cov_matrix)
    
    # # Create a random key first
    # rng_key = random.PRNGKey(0)
    # num_samples = 1000
    
    # # Sample from the priors
    # H_O_points = numpyro.sample("H_O_points", 
    #                            numpyro.distributions.Uniform(inference.Ho_bounds[0], inference.Ho_bounds[1]), 
    #                            sample_shape=(num_samples,),
    #                            rng_key=rng_key)
    
    # rng_key, subkey = random.split(rng_key)
    # Om_points = numpyro.sample("Om_points", 
    #                           numpyro.distributions.Uniform(inference.Om_bounds[0], inference.Om_bounds[1]), 
    #                           sample_shape=(num_samples,),
    #                           rng_key=subkey)
    
    # rng_key, subkey = random.split(rng_key)
    # Ok_points = numpyro.sample("Ok_points", 
    #                           numpyro.distributions.Uniform(inference.Ok_bounds[0], inference.Ok_bounds[1]), 
    #                           sample_shape=(num_samples,),
    #                           rng_key=subkey)
    
    # # plot the output of the model
    # phys_model = physics_model()  # Create an instance of the physics model
    # for i in range(len(H_O_points)):
    #     print(f"Parameters {i}: H0={H_O_points[i]:.2f}, Ωm={Om_points[i]:.2f}, Ωk={Ok_points[i]:.2f}")
    #     # Make sure z is an array
    #     z_values = jnp.array(inference.z_obs)  # Use the z values from inference
    #     comoving_distance = phys_model.calc_comoving_distance(H_O_points[i], Om_points[i], Ok_points[i], z_values)
    #     luminosity_distance = phys_model.calc_luminosity_distance(comoving_distance, Ok_points[i], z_values)
    #     mu_model = phys_model.calc_apparent_magnitude(luminosity_distance)
    #     plt.scatter(z_values, mu_model, c='blue', alpha=0.1)
    # plt.show()



