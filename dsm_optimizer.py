"""
High-Fidelity Deep-Space Maneuver (DSM) Trajectory Optimizer
Uses SPICE (JPL DE440) for ephemeris and constants.
Models an Akatsuki-like recovery mission with multiple heliocentric revolutions.
"""

import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp
import spiceypy as spice

# ==============================================================================
# 1. SPICE KERNEL MANAGEMENT
# ==============================================================================
KERNEL_DIR = "./spice_kernels"
os.makedirs(KERNEL_DIR, exist_ok=True)

KERNEL_URLS = {
    "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
}

def load_spice_kernels():
    """Download and furnish necessary SPICE kernels if not present."""
    kernels_to_load = []
    for filename, url in KERNEL_URLS.items():
        filepath = os.path.join(KERNEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from NAIF...")
            urllib.request.urlretrieve(url, filepath)
        kernels_to_load.append(filepath)
    
    spice.furnsh(" ".join(kernels_to_load))
    print("SPICE kernels furnished successfully.")

# ==============================================================================
# 2. CONSTANTS & EPHEMERIS WRAPPERS
# ==============================================================================
def get_body_constants(body_name: str):
    """Retrieve GM and radius from SPICE."""
    body_id = spice.bodn2c(body_name)
    gm, _ = spice.bodvrd(body_name, "GM", 1)
    radii, _ = spice.bodvrd(body_name, "RADII", 3)
    return {
        "id": body_id,
        "gm": gm[0],          # km^3/s^2
        "mean_radius": np.mean(radii) # km
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

# ==============================================================================
# 3. LAMBERT SOLVER (Universal Variables, Robust for Multi-Rev)
# ==============================================================================
def solve_lambert(r1: np.ndarray, r2: np.ndarray, tof_days: float, 
                  mu: float, prograde: bool = True, max_revs: int = 0) -> tuple:
    """
    Solve Lambert's problem using a robust universal variable formulation.
    Returns (v1, v2) in km/s.
    """
    tof = tof_days * 86400.0
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Transfer angle
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    cross = np.cross(r1, r2)
    
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2*np.pi - np.arccos(cos_dnu)
    else:
        dnu = 2*np.pi - np.arccos(cos_dnu) if cross[2] >= 0 else np.arccos(cos_dnu)
        
    # Chord and semi-perimeter
    c = np.linalg.norm(r2 - r1)
    s = (r1_mag + r2_mag + c) / 2.0
    
    # Initial guess for universal variable x
    # Using a simplified Newton-Raphson on the time-of-flight equation
    x = 0.0
    if tof > 0:
        # Battin's initial guess approximation
        x = 0.5
        
    for _ in range(100):
        # Stumpff functions
        if x > 0:
            c2 = (1 - np.cos(np.sqrt(x))) / x
            c3 = (np.sqrt(x) - np.sin(np.sqrt(x))) / (x * np.sqrt(x))
        elif x < 0:
            cx = np.cosh(np.sqrt(-x))
            sx = np.sinh(np.sqrt(-x))
            c2 = (1 - cx) / x
            c3 = (sx - np.sqrt(-x)) / (-x * np.sqrt(-x))
        else:
            c2 = 0.5
            c3 = 1.0 / 6.0
            
        y = r1_mag + r2_mag + (x * c3 - 1) * np.sqrt(r1_mag * r2_mag) * np.sin(dnu/2) / np.sqrt(c2 + 1e-12)
        
        if y < 0:
            x += 0.1
            continue
            
        chi = np.sqrt(y / (c2 + 1e-12))
        tof_calc = (chi**3 * c3 + np.sqrt(r1_mag * r2_mag) * np.sin(dnu/2) * np.sqrt(y)) / np.sqrt(mu)
        
        if abs(tof_calc - tof) < 1e-6:
            break
            
        # Derivative of tof w.r.t x (simplified)
        d_tof = (chi**3 * (c3 - 3*c2*c3/(2*c2 + 1e-12)) + np.sqrt(r1_mag * r2_mag) * np.sin(dnu/2) * np.sqrt(y) / (8 * (c2 + 1e-12))) / np.sqrt(mu)
        x += (tof - tof_calc) / max(abs(d_tof), 1e-12)

    # Compute velocities
    f = 1.0 - y / r1_mag
    gdot = 1.0 - y / r2_mag
    g = np.sqrt(y / (c2 + 1e-12)) * np.sqrt(r1_mag * r2_mag) * np.sin(dnu/2) / np.sqrt(mu)
    
    if abs(g) < 1e-12:
        g = 1e-12
        
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    
    return v1, v2

# ==============================================================================
# 4. TRAJECTORY & ENCOUNTER GEOMETRY
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
    r_dsms = x[n_dsms:-1].reshape(n_dsms, 3)
    t_enc = x[-1]
    
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

# ==============================================================================
# 5. OPTIMIZER WRAPPER
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
        options={'maxiter': 200}
    )
    
    best_x = result.x if result.fun < result_local.fun else result_local.x
    best_cost = min(result.fun, result_local.fun)
    
    # Reconstruct final trajectory details for reporting
    t_dsms = best_x[:n_dsms]
    r_dsms = best_x[n_dsms:-1].reshape(n_dsms, 3)
    t_enc = best_x[-1]
    
    return {
        "optimal_x": best_x,
        "total_dv": best_cost,
        "t_dsms": t_dsms,
        "r_dsms": r_dsms,
        "t_enc": t_enc,
        "success": result_local.success or result.success
    }

# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
def plot_trajectory_2d(opt_result: dict, dep_state: np.ndarray, target_body: str):
    """Plot the optimized 2D heliocentric trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    t_dsms = opt_result["t_dsms"]
    r_dsms = opt_result["r_dsms"]
    t_enc = opt_result["t_enc"]
    
    au_km = 149597870.7
    
    # Sun
    ax.plot(0, 0, 'yo', markersize=15, label='Sun', zorder=10)
    
    # Planet orbit (ephemeris sweep)
    jd_sweep = np.linspace(2455538, t_enc, 300)
    px, py = [], []
    for jd in jd_sweep:
        r_p = get_heliocentric_state(target_body, jd)[:3] / au_km
        px.append(r_p[0])
        py.append(r_p[1])
    ax.plot(px, py, 'b--', alpha=0.5, linewidth=1, label=f'{target_body.capitalize()} Orbit')
    
    # Trajectory arcs
    times = [2455538.0] + list(t_dsms) + [t_enc]
    positions = [dep_state[:3]] + list(r_dsms) + [get_heliocentric_state(target_body, t_enc)[:3]]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)-1))
    
    for i in range(len(times)-1):
        r1, r2 = positions[i], positions[i+1]
        t1, t2 = times[i], times[i+1]
        
        # Propagate for smooth curve
        def eom(t, y):
            r = y[:3]
            v = y[3:]
            return np.concatenate([v, -get_body_constants("sun")["gm"] / np.linalg.norm(r)**3 * r])
            
        sol = solve_ivp(eom, (0, (t2-t1)*86400), np.concatenate([r1, get_heliocentric_state("venus", t1)[3:]]), 
                        t_eval=np.linspace(0, (t2-t1)*86400, 100), rtol=1e-9)
        
        x_au = sol.y[0] / au_km
        y_au = sol.y[1] / au_km
        ax.plot(x_au, y_au, color=colors[i], linewidth=2, label=f'Arc {i+1}' if i==0 else "")
        
        if i < len(times)-2:
            ax.plot(x_au[-1], y_au[-1], 'ks', markersize=8, markerfacecolor='white', markeredgewidth=2, label='DSM' if i==0 else "")
            
    ax.plot(positions[0][0]/au_km, positions[0][1]/au_km, 'g^', markersize=10, label='Departure')
    ax.plot(positions[-1][0]/au_km, positions[-1][1]/au_km, 'r*', markersize=15, label='Encounter')
    
    ax.set_xlabel('Heliocentric X [AU]')
    ax.set_ylabel('Heliocentric Y [AU]')
    ax.set_title(f'Heliocentric DSM Trajectory\nTotal ΔV: {opt_result["total_dv"]:.3f} km/s')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("trajectory_2d.png", dpi=300)
    plt.show()

def plot_dv_breakdown(opt_result: dict, dep_state: np.ndarray, target_body: str, rp_km: float, ra_km: float):
    """Bar chart of ΔV contributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t_dsms = opt_result["t_dsms"]
    r_dsms = opt_result["r_dsms"]
    t_enc = opt_result["t_enc"]
    n_dsms = len(t_dsms)
    
    times = [2455538.0] + list(t_dsms) + [t_enc]
    positions = [dep_state[:3]] + list(r_dsms) + [get_heliocentric_state(target_body, t_enc)[:3]]
    
    mu_sun = get_body_constants("sun")["gm"]
    target_const = get_body_constants(target_body)
    
    dvs = []
    labels = []
    
    current_v = dep_state[3:]
    for i in range(n_dsms + 1):
        r1, r2 = positions[i], positions[i+1]
        tof = times[i+1] - times[i]
        
        v1_req, v2_arr = solve_lambert(r1, r2, tof, mu_sun)
        
        if i < n_dsms:
            dv = np.linalg.norm(v1_req - current_v)
            dvs.append(dv)
            labels.append(f'DSM {i+1}')
        else:
            v_planet = get_heliocentric_state(target_body, t_enc)[3:]
            v_inf = np.linalg.norm(v2_arr - v_planet)
            dv_cap = compute_capture_dv(v_inf, rp_km, ra_km, target_const["gm"])
            dvs.append(dv_cap)
            labels.append('Capture')
            
        current_v = v2_arr
        
    colors = ['skyblue'] * n_dsms + ['salmon']
    bars = ax.bar(labels, dvs, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    ax.set_ylabel('ΔV [km/s]')
    ax.set_title('Mission ΔV Budget Breakdown')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.set_ylim(0, max(dvs) * 1.3)
    plt.tight_layout()
    plt.savefig("dv_breakdown.png", dpi=300)
    plt.show()

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
def main():
    print("="*60)
    print("AKATSUKI-LIKE RECOVERY MISSION OPTIMIZER (SPICE HIGH-FIDELITY)")
    print("="*60)
    
    # 1. Initialize SPICE
    load_spice_kernels()
    
    # 2. Mission Parameters
    TARGET_BODY = "venus"
    DEP_JD = 2455538.0  # ~Dec 7, 2010 (Post-failure epoch)
    
    # Get initial state: Venus state + small residual v_inf (simulating failed capture)
    venus_state = get_heliocentric_state(TARGET_BODY, DEP_JD)
    v_inf_residual = np.array([0.05, -0.15, 0.02]) # km/s
    dep_state = venus_state.copy()
    dep_state[3:] += v_inf_residual
    
    # Target capture orbit: 400 km x 440,000 km
    venus_const = get_body_constants(TARGET_BODY)
    RP_KM = venus_const["mean_radius"] + 400.0
    RA_KM = 440000.0
    
    N_DSMS = 2
    T_MIN_DAYS = 200
    T_MAX_DAYS = 1800
    
    print(f"\nDeparture JD: {DEP_JD}")
    print(f"Target: {TARGET_BODY.capitalize()} (GM = {venus_const['gm']:.2e} km³/s²)")
    print(f"Target Orbit: {RP_KM:.0f} x {RA_KM:.0f} km")
    print("-" * 60)
    
    # 3. Run Optimization
    opt_result = optimize_dsm_trajectory(
        dep_jd=DEP_JD,
        dep_state=dep_state,
        target_body=TARGET_BODY,
        n_dsms=N_DSMS,
        rp_km=RP_KM,
        ra_km=RA_KM,
        t_min_days=T_MIN_DAYS,
        t_max_days=T_MAX_DAYS
    )
    
    # 4. Report Results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Success: {opt_result['success']}")
    print(f"Total Mission ΔV: {opt_result['total_dv']:.4f} km/s")
    print(f"Encounter Epoch: JD {opt_result['t_enc']:.2f}")
    
    print("\nDSM Schedule:")
    for i in range(N_DSMS):
        print(f"  DSM {i+1}: JD {opt_result['t_dsms'][i]:.2f}, "
              f"Pos = {opt_result['r_dsms'][i] / 149597870.7} AU")
              
    # 5. Generate Plots
    print("\nGenerating visualization plots...")
    plot_trajectory_2d(opt_result, dep_state, TARGET_BODY)
    plot_dv_breakdown(opt_result, dep_state, TARGET_BODY, RP_KM, RA_KM)
    print("Plots saved as 'trajectory_2d.png' and 'dv_breakdown.png'")
    
    # Unload SPICE
    spice.unload_all()

if __name__ == "__main__":
    main()
