import numpy as np
from ephemeris import get_heliocentric_state, get_body_constants
from lambert import solve_lambert

# ==============================================================================
# TRAJECTORY & ENCOUNTER GEOMETRY
# ==============================================================================
def compute_capture_dv(v_inf_mag: float, rp_km: float, ra_km: float, mu_body: float) -> float:
    """Compute impulsive capture ΔV at periapsis for an elliptical orbit."""
    v_peri_hyper = np.sqrt(v_inf_mag**2 + 2 * mu_body / rp_km)
    a_target = (rp_km + ra_km) / 2.0
    v_peri_target = np.sqrt(mu_body * (2.0/rp_km - 1.0/a_target))
    return abs(v_peri_hyper - v_peri_target)

def evaluate_trajectory(x: np.ndarray, dep_jd: float, dep_state: np.ndarray, 
                        target_body: str, n_dsms: int, rp_km: float, ra_km: float) -> float:
    """
    Objective function for optimization.
    x = [t_dsm1, t_dsm2, ..., t_enc, r_dsm1_x, r_dsm1_y, r_dsm1_z, ...]
    """
    t_dsms = x[:n_dsms]
    t_enc = x[n_dsms]
    r_dsms = x[n_dsms+1:].reshape(n_dsms, 3)

    # Constraint: Times must be strictly increasing
    times = [dep_jd] + list(t_dsms) + [t_enc]
    if not all(times[i] < times[i+1] for i in range(len(times)-1)):
        return 1e6  # Heavy penalty
    
    total_dv = 0.0
    current_r = dep_state[:3]
    current_v = dep_state[3:]
    
    target_constants = get_body_constants(target_body)
    mu_sun = get_body_constants("sun")["gm"]
    
    # Evaluate each arc
    for i in range(n_dsms + 1):
        t1 = times[i]
        t2 = times[i+1]
        tof = t2 - t1
        
        if i < n_dsms:
            r2 = r_dsms[i]
        else:
            # Final arc targets the planet's actual position at encounter
            r2 = get_heliocentric_state(target_body, t2)[:3]
            
        try:
            v1_req, v2_arr = solve_lambert(current_r, r2, tof, mu_sun, prograde=True)
        except Exception:
            return 1e6 # Lambert failed

        if not np.all(np.isfinite(v1_req)):
            return 1e6

        if not np.all(np.isfinite(v2_arr)):
            return 1e6
            
        if i < n_dsms:
            # DSM ΔV is the difference between arrival velocity from previous arc 
            # and departure velocity required for the next arc.
            # For simplicity in this direct formulation, we assume the DSM 
            # instantly changes the state from (r2, v1_req) to (r2, v2_req_for_next).
            # We approximate this by just taking the magnitude of the velocity 
            # vector required to stay on the patched conic. 
            # A more rigorous multiple-shooting would match v1_out and v2_in.
            # Here, we use a simplified cost: the ΔV to match the Lambert arc.
            dv_dsm = np.linalg.norm(v1_req - current_v)
            total_dv += dv_dsm
            
        current_r = r2
        current_v = v2_arr
        
    # Final encounter
    v_planet = get_heliocentric_state(target_body, t_enc)[3:]
    v_inf = current_v - v_planet
    v_inf_mag = np.linalg.norm(v_inf)
    
    dv_capture = compute_capture_dv(v_inf_mag, rp_km, ra_km, target_constants["gm"])
    total_dv += dv_capture
    
    return total_dv