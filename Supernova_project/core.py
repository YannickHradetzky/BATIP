import jax.numpy as jnp
from scipy.integrate import quad
from jax import vmap
# Constants
c = 299792.458  # speed of light in km/s

def H(z, H0, Om):
    """Hubble parameter at redshift z"""
    return H0 * jnp.sqrt(Om * (1 + z)**3 + (1 - Om))  # Assuming flat universe

def integrand(z, Om, H0):
    """Integrand for the comoving distance"""
    return c / H(z, H0, Om)

def luminosity_distance(z, Om, H0):
    """Calculate luminosity distance in Mpc"""
    # Integrate to get comoving distance
    chi, _ = quad(lambda z_: integrand(z_, Om, H0), 0, z)
    # Convert to luminosity distance
    return (1 + z) * chi

def distance_modulus(z, Om, H0):
    """Calculate distance modulus"""
    dL = luminosity_distance(z, Om, H0)
    return 5 * jnp.log10(dL) + 25  # Factor of 25 converts from Mpc to pc

distance_modulus_vec = vmap(distance_modulus, in_axes=(0, None, None))