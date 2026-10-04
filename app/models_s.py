import numpy as np
import matplotlib.tri as tri

# =====================================================================
# 1. BISHOP LIMIT EQUILIBRIUM MODEL (EXISTING CODE - UNTOUCHED)
# =====================================================================
def calculate_slope_stability(xc, yc, R, sensor_u_kpa, kh=0.0, gamma=18, gamma_w=9.81, c=15, phi=25, custom_phreatic_fn=None):
    """
    Bishop Stability Analysis with Dupuit Parabola or Custom FEM Phreatic Line,
    and Pseudo-static Seismic kh.
    """
    # Dam Geometry
    dx = np.array([0, 40, 70, 100, 130, 200])
    dy = np.array([10, 10, 45, 45, 14, 14])
    
    # 1. DEFINE THE PHREATIC LINE (CUSTOM FEM OR DUPUIT PARABOLA)
    if custom_phreatic_fn is not None:
        get_phreatic_y = custom_phreatic_fn
    else:
        h_at_sensor = sensor_u_kpa / gamma_w
        y_at_sensor = 10 + h_at_sensor
        x_toe, y_toe = 40, 10
        k = (y_at_sensor - y_toe)**2 / max(1e-3, (80 - x_toe))

        def get_phreatic_y(x):
            if x < x_toe: return y_toe
            return np.sqrt(max(0, k * (x - x_toe))) + y_toe

    # 2. FIND INTERSECTIONS
    x_scan = np.linspace(xc - R + 0.01, xc + R - 0.01, 2000)
    y_dam_scan = np.interp(x_scan, dx, dy)
    y_circ_scan = yc - np.sqrt(R**2 - (x_scan - xc)**2)
    
    diff = y_dam_scan - y_circ_scan
    abs_diff = np.signbit(diff)
    sign_changes = np.where(abs_diff[:-1] != abs_diff[1:])[0]
    
    if len(sign_changes) < 2:
        return 0.0, [], None, [], 0.0, 0.0

    idx_start, idx_end = sign_changes[0], sign_changes[-1]
    x_start, x_end = x_scan[idx_start], x_scan[idx_end]
    
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
        
        y_center = y_bot + (h_slice / 2)
        hi = yc - y_center
        
        y_water = get_phreatic_y(x_mid)
        h_water = y_water - y_bot
        u_slice = h_water * gamma_w if h_water > 0 else 0

        W = h_slice * b * gamma
        alpha_rad = np.arcsin((x_mid - xc) / R) 
        
        slices.append({
            'W': W, 'alpha_rad': alpha_rad, 'b': b, 'u': u_slice, 
            'x_mid': x_mid, 'h': h_slice, 'y_bot': y_bot, 'hi': hi
        })

    # 4. BISHOP SOLVER
    fs = 1.2
    convergence_history = []
    for i in range(25):
        convergence_history.append(fs)
        num, den = 0, 0
        for s in slices:
            a_rad = s['alpha_rad']
            
            static_moment = s['W'] * np.sin(a_rad)
            seismic_moment = abs(kh * s['W'] * s['hi'] / R)
            den += (static_moment + seismic_moment)
            
            m_alpha = np.cos(a_rad) + (np.sin(a_rad) * np.tan(phi_rad) / fs)
            if m_alpha < 0.1: m_alpha = 0.1

            effective_weight = s['W'] - (s['u'] * s['b'])
            resisting = (c * s['b'] + max(0, effective_weight) * np.tan(phi_rad)) / m_alpha
            num += resisting

        if abs(den) < 1e-5:
            new_fs = 10.0
        else:
            new_fs = num / den

        convergence_history.append(new_fs)
        if abs(new_fs - fs) < 0.001:
            break
        fs = new_fs

    if fs > 50:
        return 50.0, slices, (w_x, w_y), convergence_history, num, den
        
    return round(fs, 3), slices, (w_x, w_y), convergence_history, num, den


# =====================================================================
# 2. 2D UNCONFINED STATIONARY DARCY FEM SOLVER (NEW ADDITION)
# =====================================================================
def solve_darcy_fem(L_bottom=200, L_top=30, H_dam=35, h_pool=30.0, k_sat=1e-5, nx=30, ny=15, gamma_w=9.81):
    """
    2D Steady-State Unconfined Darcy Seepage Solver using Finite Elements (T3 Elements).
    Returns mesh geometry, hydraulic heads (h), pressure heads (psi), pore pressure (P),
    and an interpolated phreatic surface function y_phreatic(x).
    """
    # 1. Generate Structured Grid over Trapezoidal Tailings Geometry
    # Slope boundaries: Crest (left and right slopes)
    x_left_crest = 40.0
    x_right_crest = x_left_crest + L_top
    y_base = 10.0
    y_top = y_base + H_dam
    
    x_coords = []
    y_coords = []
    
    for i in range(ny + 1):
        eta = i / ny
        y = y_base + eta * H_dam
        
        # Interpolate domain boundaries at elevation y
        x_min = 0.0
        x_max = L_bottom
        
        x_line = np.linspace(x_min, x_max, nx + 1)
        for x in x_line:
            # Mask to dam outer bounds
            y_surface = np.interp(x, [0, 40, 40 + L_top, L_bottom], [y_base, y_base, y_top, y_base])
            if y <= y_surface:
                x_coords.append(x)
                y_coords.append(y)

    nodes = np.column_stack((x_coords, y_coords))
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1])
    elements = triangulation.triangles
    num_nodes = len(nodes)
    
    # 2. Global Stiffness Matrix Construction
    K_global = np.zeros((num_nodes, num_nodes))
    
    for elem in elements:
        pts = nodes[elem]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        
        # Element Area
        two_A = (x2*y3 - x3*y2) - (x1*y3 - x3*y1) + (x1*y2 - x2*y1)
        Area = 0.5 * abs(two_A)
        if Area < 1e-9:
            continue
            
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        # Element Conductance Matrix (Isotropic)
        K_elem = (k_sat / (4.0 * Area)) * (np.outer(b, b) + np.outer(c, c))
        
        for i_local in range(3):
            for j_local in range(3):
                K_global[elem[i_local], elem[j_local]] += K_elem[i_local, j_local]

    # 3. Apply Boundary Conditions
    # Reservoir pool boundary (Left upstream face: h = y_base + h_pool)
    # Downstream seepage face: h = z
    h_fem = np.zeros(num_nodes)
    prescribed = np.zeros(num_nodes, dtype=bool)
    
    h_upstream = y_base + h_pool
    
    for idx, (x, y) in enumerate(nodes):
        # Upstream Pool Boundary
        if x <= 40 and y <= h_upstream:
            prescribed[idx] = True
            h_fem[idx] = h_upstream
        # Downstream Toe / Drainage Face
        elif x >= (40 + L_top) and y <= (y_base + 4.0):
            prescribed[idx] = True
            h_fem[idx] = y

    # 4. Solve System K * h = F
    F_global = np.zeros(num_nodes)
    free_dofs = np.where(~prescribed)[0]
    prescribed_dofs = np.where(prescribed)[0]
    
    if len(free_dofs) > 0:
        F_global[free_dofs] -= K_global[np.ix_(free_dofs, prescribed_dofs)] @ h_fem[prescribed_dofs]
        h_fem[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])

    # 5. Pressure Head & Pore Pressure Field
    psi = h_fem - nodes[:, 1]  # psi = h - z
    P_kpa = np.maximum(0, psi * gamma_w)  # P = gamma_w * psi

    # 6. Extract Phreatic Line (psi = 0 contour)
    x_phreatic = np.linspace(40, L_bottom, 100)
    y_phreatic_vals = []
    
    for x_q in x_phreatic:
        # Find nodes close to this x_q
        mask = np.abs(nodes[:, 0] - x_q) < (L_bottom / nx)
        if np.any(mask):
            sub_nodes = nodes[mask]
            sub_psi = psi[mask]
            # Interpolate zero crossing for psi along y
            if np.min(sub_psi) <= 0 <= np.max(sub_psi):
                y_zero = np.interp(0, sub_psi[np.argsort(sub_nodes[:, 1])], np.sort(sub_nodes[:, 1]))
                y_phreatic_vals.append(y_zero)
            elif np.all(sub_psi > 0):
                y_phreatic_vals.append(np.max(sub_nodes[:, 1]))
            else:
                y_phreatic_vals.append(y_base)
        else:
            y_phreatic_vals.append(y_base)

    def fem_phreatic_fn(x):
        return np.interp(x, x_phreatic, y_phreatic_vals, left=y_base, right=y_base)

    return {
        "triangulation": triangulation,
        "nodes": nodes,
        "elements": elements,
        "h_fem": h_fem,
        "psi": psi,
        "P_kpa": P_kpa,
        "x_phreatic": x_phreatic,
        "y_phreatic": y_phreatic_vals,
        "phreatic_fn": fem_phreatic_fn
    }
