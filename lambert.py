import numpy as np

# ==============================================================================
# LAMBERT SOLVER (Universal Variables, Robust for Multi-Rev)
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