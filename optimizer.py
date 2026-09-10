import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
from ephemeris import get_heliocentric_state, get_body_constants
from trajectory import evaluate_trajectory

# ==============================================================================
# OPTIMIZER WRAPPER
# ==============================================================================
def optimize_dsm_trajectory(dep_jd: float, dep_state: np.ndarray, target_body: str, 
                            n_dsms: int, rp_km: float, ra_km: float, 
                            t_min_days: float, t_max_days: float) -> dict:
    """
    Run global optimization to find optimal DSM epochs and positions.
    """
    print(f"Optimizing trajectory with {n_dsms} DSM(s)...")
    
    # Decision variables: [t_dsm1, ..., t_enc, r_dsm1_x, r_dsm1_y, r_dsm1_z, ...]
    n_vars = n_dsms + 1 + (n_dsms * 3)
    
    # Bounds
    bounds = []
    for _ in range(n_dsms):
        bounds.append((dep_jd + t_min_days, dep_jd + t_max_days)) # t_dsm
    bounds.append((dep_jd + t_min_days + 100, dep_jd + t_max_days + 500)) # t_enc
    
    # Position bounds (roughly 0.5 to 1.5 AU in km)
    au_km = 149597870.7
    for _ in range(n_dsms * 3):
        bounds.append((-1.5 * au_km, 1.5 * au_km))
        
    # Initial guess: linear time spacing, positions near Venus orbit
    x0 = np.zeros(n_vars)
    t_guess = np.linspace(dep_jd + t_min_days, dep_jd + t_max_days, n_dsms + 1)
    x0[:n_dsms+1] = t_guess
    
    venus_r = get_heliocentric_state("venus", dep_jd)[:3]
    for i in range(n_dsms):
        x0[n_dsms + 1 + i*3 : n_dsms + 1 + (i+1)*3] = venus_r * (0.8 + 0.4 * np.random.rand())

    # Run Differential Evolution (robust for multimodal astrodynamics problems)
    result = differential_evolution(
        evaluate_trajectory,
        bounds,
        args=(dep_jd, dep_state, target_body, n_dsms, rp_km, ra_km),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        disp=True,
        seed=42
    )
    
    # Refine with local optimizer
    result_local = minimize(
        evaluate_trajectory,
        result.x,
        args=(dep_jd, dep_state, target_body, n_dsms, rp_km, ra_km),
        method='SLSQP',
        options={'maxiter': 200,         
                 'ftol': 1e-8,
                 'disp': True
                }
    )

    print(result)
    print(result_local)
    
    best_x = result.x if result.fun < result_local.fun else result_local.x
    best_cost = min(result.fun, result_local.fun)
    
    # Reconstruct final trajectory details for reporting
    '''
    t_dsms = best_x[:n_dsms]
    r_dsms = best_x[n_dsms:-1].reshape(n_dsms, 3)
    t_enc = best_x[-1]
    '''

    t_dsms = best_x[:n_dsms]
    t_enc = best_x[n_dsms]
    r_dsms = best_x[n_dsms + 1:].reshape(n_dsms, 3)
    
    return {
        "optimal_x": best_x,
        "total_dv": best_cost,
        "t_dsms": t_dsms,
        "r_dsms": r_dsms,
        "t_enc": t_enc,
        "success": result_local.success or result.success
    }
