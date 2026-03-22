import numpy as np

def calculate_slope_stability(xc, yc, R, sensor_u_kpa, kh=0.0, gamma=18, gamma_w=9.81, c=15, phi=25):
    """
    Bishop Stability Analysis with Dupuit Parabola and Pseudo-static Seismic kh.
    """
    # Dam Geometry
    dx = np.array([0, 40, 70, 100, 130, 200])
    dy = np.array([10,10, 45, 45, 14, 14])
    
    # 1. DEFINE THE DUPUIT PARABOLA
    h_at_sensor = sensor_u_kpa / gamma_w
    y_at_sensor = 10 + h_at_sensor
    x_toe, y_toe = 40, 10
    k = (y_at_sensor - y_toe)**2 / (80 - x_toe)

    def get_phreatic_y(x):
        if x < x_toe: return y_toe
        return np.sqrt(max(0, k * (x - x_toe))) + y_toe

    # 2. FIND INTERSECTIONS
    x_scan = np.linspace(xc - R + 0.01, xc + R - 0.01, 1000) # 500
    y_dam_scan = np.interp(x_scan, dx, dy)
    y_circ_scan = yc - np.sqrt(R**2 - (x_scan - xc)**2)
    
    diff = y_dam_scan - y_circ_scan
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) < 2:
        return None, None, None
    
    x_start, x_end = x_scan[sign_changes[0]], x_scan[sign_changes[-1]]
    
    # 3. SLICES
    num_slices = 30
    slice_edges = np.linspace(x_start, x_end, num_slices + 1)
    b = (x_end - x_start) / num_slices
    phi_rad = np.radians(phi)
    
    slices = []
    w_x = np.linspace(40, 130, 100)
    w_y = [get_phreatic_y(x) for x in w_x]

    for i in range(num_slices):
        x_mid = (slice_edges[i] + slice_edges[i+1]) / 2
        y_top = np.interp(x_mid, dx, dy)
        y_bot = yc - np.sqrt(R**2 - (x_mid - xc)**2)
        h_slice = max(0, y_top - y_bot)
        
        # Calculate Vertical Midpoint of the slice for the Seismic Lever Arm (hi)
        y_center = y_bot + (h_slice / 2)
        hi = yc - y_center  # Perpendicular distance to center O for horizontal force
        
        y_water = get_phreatic_y(x_mid)
        h_water = y_water - y_bot
        u_slice = h_water * gamma_w if h_water > 0 else 0

        W = h_slice * b * gamma
        alpha_rad = np.arcsin((xc - x_mid) / R) 
        
        slices.append({
            'W': W, 'alpha_rad': alpha_rad, 'b': b, 'u': u_slice, 
            'x_mid': x_mid, 'h': h_slice, 'y_bot': y_bot, 'hi': hi
        })

    # 4. BISHOP SOLVER (Updated for kh)
    fs = 1.2
    for i in range(25):
        num, den = 0, 0
        for s in slices:
            a_rad = s['alpha_rad']
            
            # Driving Force = Static Moment + Seismic Moment (Normalized by R)
            static_moment = s['W'] * np.sin(a_rad)
            seismic_moment = abs(kh * s['W'] * s['hi'] / R)
            den += (static_moment + seismic_moment)
            
            m_alpha = np.cos(a_rad) + (np.sin(a_rad) * np.tan(phi_rad) / fs)
            # GUARDRAIL: If m_alpha is too small or negative, the physics breaks.
            # We force it to a minimum positive value (0.1) to maintain stability.
            if m_alpha < 0.1: m_alpha = 0.1

            # Resisting Force
            # We also ensure (W - ub) doesn't create negative friction (suction)
            # unless your model specifically supports it.
            effective_weight = s['W'] - (s['u'] * s['b'])
            resisting = (c * s['b'] + max(0, effective_weight) * np.tan(phi_rad)) / m_alpha
            num += resisting

        # Avoid division by zero if circle is symmetric
        if abs(den) < 1e-5:
            new_fs = 10.0 # Effectively "Infinite" safety if there is no driving force
        else:
            new_fs = num / den

        # Convergence Check
        if abs(new_fs - fs) < 0.001:
            break
        fs = new_fs

    # Final logic: if FS is still unrealistic, return a special value
    if fs < 0 or fs > 50:
        return 0.0, slices, (w_x, w_y) # 0.0 signals a non-physical geometry
        
    return round(fs, 3), slices, (w_x, w_y)
