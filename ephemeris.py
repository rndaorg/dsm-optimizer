import numpy as np
import spiceypy as spice

# ==============================================================================
# CONSTANTS & EPHEMERIS WRAPPERS
# ==============================================================================
def get_body_constants(body_name: str):
    """Retrieve GM and radius from SPICE."""
    body_id = spice.bodn2c(body_name)
    gm = spice.bodvrd(body_name, "GM", 1)
    radii = spice.bodvrd(body_name, "RADII", 3)

    return {
        "id": body_id,
        "gm": gm[1][0],          # km^3/s^2
        "mean_radius": np.mean(radii[1]) # km
    }

def get_heliocentric_state(body_name: str, jd: float) -> np.ndarray:
    """
    Get heliocentric state [x, y, z, vx, vy, vz] in J2000 frame.
    """
    et = spice.str2et(f"{jd} JD")
    # Sun ID is 10. We get state of body relative to Sun.
    state, lt = spice.spkgeo(spice.bodn2c(body_name), et, "J2000", 10)
    return np.array(state) # [km, km/s]

def jd_to_et(jd: float) -> float:
    return spice.str2et(f"{jd} JD")