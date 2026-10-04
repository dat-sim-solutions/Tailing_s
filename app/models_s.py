import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


# =====================================================================
# 1. FEM UNCONFINED SEEPAGE FUNCTIONS
# =====================================================================
def quad_element_matrices_unconfined(x_e, y_e, K_sat, h_elem_nodes):
    """Computes quadrilateral element stiffness matrix with unconfined cutoff for unsaturated nodes."""
    gauss_pts = [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
    weights = [1.0, 1.0]

    Ke = np.zeros((4, 4))

    for xi, w_xi in zip(gauss_pts, weights):
        for eta, w_eta in zip(gauss_pts, weights):
            # Shape functions & derivatives
            N = 0.25 * np.array(
                [
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ]
            )
            dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])

            # Evaluate elevation (z) and head (h) at current Gauss point
            z_gauss = np.dot(N, y_e)
            h_gauss = np.dot(N, h_elem_nodes)
            psi_gauss = h_gauss - z_gauss  # Pressure head psi = h - z

            # Relative permeability cutoff (unsaturated zone)
            kr = 1.0 if psi_gauss >= 0.0 else 1e-4

            J = np.zeros((2, 2))
            J[0, 0] = np.dot(dN_dxi, x_e)
            J[0, 1] = np.dot(dN_dxi, y_e)
            J[1, 0] = np.dot(dN_deta, x_e)
            J[1, 1] = np.dot(dN_deta, y_e)

            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            dN_dx_dy = invJ @ np.vstack((dN_dxi, dN_deta))

            weight = w_xi * w_eta * detJ
            Ke += (K_sat * kr) * (dN_dx_dy.T @ dN_dx_dy) * weight

    return Ke


def run_post_rain_seepage():
    """
    Solves steady-state unconfined seepage inside the inverted trapezoid domain
    under post-rain conditions with a standing water pool boundary.
    """
    Lx_bot, Lx_top, H_dam = 300.0, 500.0, 100.0
    K_sat = 1e-5
    h_rain_pool = 102.0  # 100m dam height + 2m standing water after rain
    nx, ny = 40, 20

    # Build Inverted Trapezoid Mesh
    L_slope = Lx_top - Lx_bot
    xi_grid = np.linspace(0, 1, nx + 1)
    eta_grid = np.linspace(0, 1, ny + 1)

    node_coords = []
    node_id_map = np.zeros((ny + 1, nx + 1), dtype=int)
    current_id = 0

    for j, eta in enumerate(eta_grid):
        y_val = eta * H_dam
        x_left = -eta * L_slope
        x_right = Lx_bot

        for i, xi in enumerate(xi_grid):
            x_val = x_left + xi * (x_right - x_left)
            node_coords.append([x_val, y_val])
            node_id_map[j, i] = current_id
            current_id += 1

    node_coords = np.array(node_coords)
    num_nodes = len(node_coords)

    elements = []
    for j in range(ny):
        for i in range(nx):
            elements.append(
                [
                    node_id_map[j, i],
                    node_id_map[j, i + 1],
                    node_id_map[j + 1, i + 1],
                    node_id_map[j + 1, i],
                ]
            )

    # Identify Boundary Nodes
    top_nodes = [node_id_map[ny, i] for i in range(nx + 1)]
    sloping_left_nodes = [node_id_map[j, 0] for j in range(ny + 1)]

    # Pool boundary: Only nodes on the top surface where x > 50m (impoundment pool area)
    pool_nodes = [n for n in top_nodes if node_coords[n, 0] >= 50.0]

    # Initial head guess
    h_fem = node_coords[:, 1] + 5.0

    # Picard Iterations
    for it in range(40):
        h_old = h_fem.copy()

        fixed_nodes = []
        fixed_vals = {}

        # Fix pool head
        for n in pool_nodes:
            fixed_nodes.append(n)
            fixed_vals[n] = h_rain_pool

        # Seepage face on sloping wall (h = z if saturated)
        for n in sloping_left_nodes:
            z_n = node_coords[n, 1]
            if h_fem[n] >= z_n or n == node_id_map[0, 0]:
                fixed_nodes.append(n)
                fixed_vals[n] = z_n

        fixed_nodes = list(set(fixed_nodes))
        free_nodes = [n for n in range(num_nodes) if n not in fixed_nodes]

        K_global = np.zeros((num_nodes, num_nodes))
        for elem in elements:
            Ke = quad_element_matrices_unconfined(
                node_coords[elem, 0], node_coords[elem, 1], K_sat, h_fem[elem]
            )
            for i in range(4):
                for j in range(4):
                    K_global[elem[i], elem[j]] += Ke[i, j]

        RHS = np.zeros(num_nodes)
        for n in fixed_nodes:
            RHS[free_nodes] -= K_global[free_nodes, n] * fixed_vals[n]

        h_free = np.linalg.solve(
            K_global[np.ix_(free_nodes, free_nodes)], RHS[free_nodes])

        for n in fixed_nodes:
            h_fem[n] = fixed_vals[n]
        h_fem[free_nodes] = h_free

        if np.max(np.abs(h_fem - h_old)) < 1e-3:
            break

    return node_coords, elements, h_fem


def solve_darcy_fem(h_pool=102.0, k_sat=1e-5):
    """
    Wrapper function maintaining compatibility with external function calls and
    returning extracted phreatic line data and triangulation objects.
    """
    node_coords, elements, h_fem = run_post_rain_seepage()
    psi = h_fem - node_coords[:, 1] # z_coords

    # Extract Phreatic Line (psi = 0)
    x_min, x_max = np.min(node_coords[:, 0]), np.max(node_coords[:, 0])
    x_phreatic = np.linspace(x_min, x_max, 100)
    y_phreatic_vals = []

    for x_q in x_phreatic:
        mask = np.abs(node_coords[:, 0] - x_q) < 10.0
        if np.any(mask):
            sub_nodes = node_coords[mask]
            sub_psi = psi[mask]
            if np.min(sub_psi) <= 0 <= np.max(sub_psi):
                sort_idx = np.argsort(sub_nodes[:, 1])
                y_zero = np.interp(0, sub_psi[sort_idx], sub_nodes[sort_idx, 1])
                y_phreatic_vals.append(y_zero)
            elif np.all(sub_psi > 0):
                y_phreatic_vals.append(np.max(sub_nodes[:, 1]))
            else:
                y_phreatic_vals.append(0.0)
        else:
            y_phreatic_vals.append(0.0)

    def fem_phreatic_fn(x):
        return np.interp(x, x_phreatic, y_phreatic_vals, left=0.0, right=0.0)

    triangulation = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    gamma_w = 9.81
    P_kpa = np.maximum(0, psi * gamma_w)

    return {
        "triangulation": triangulation,
        "nodes": node_coords,
        "elements": elements,
        "h_fem": h_fem,
        "psi": psi,
        "P_kpa": P_kpa,
        "x_phreatic": x_phreatic,
        "y_phreatic": y_phreatic_vals,
        "phreatic_fn": fem_phreatic_fn,
    }


# =====================================================================
# 2. BISHOP SLOPE STABILITY ANALYSIS MODEL
# =====================================================================
def calculate_slope_stability(
    xc, yc, R, sensor_u_kpa, kh=0.0, gamma=18, gamma_w=9.81, c=15, phi=25, custom_phreatic_fn=None
):
    """
    Bishop Stability Analysis with Dupuit Parabola or FEM Phreatic Line.
    """
    dx = np.array([0, 40, 70, 100, 130, 200])
    dy = np.array([10, 10, 45, 45, 14, 14])

    if custom_phreatic_fn is not None:
        get_phreatic_y = custom_phreatic_fn
    else:
        h_at_sensor = sensor_u_kpa / gamma_w
        y_at_sensor = 10 + h_at_sensor
        x_toe, y_toe = 40, 10
        k = (y_at_sensor - y_toe) ** 2 / (80 - x_toe)

        def get_phreatic_y(x):
            if x < x_toe:
                return y_toe
            return np.sqrt(max(0, k * (x - x_toe))) + y_toe

    x_scan = np.linspace(xc - R + 0.01, xc + R - 0.01, 2000)
    y_dam_scan = np.interp(x_scan, dx, dy)
    y_circ_scan = yc - np.sqrt(R**2 - (x_scan - xc) ** 2)

    diff = y_dam_scan - y_circ_scan
    abs_diff = np.signbit(diff)
    sign_changes = np.where(abs_diff[:-1] != abs_diff[1:])[0]

    if len(sign_changes) < 2:
        return 0.0, [], (np.array([]), np.array([])), [], 0.0, 0.0

    idx_start, idx_end = sign_changes[0], sign_changes[-1]
    x_start, x_end = x_scan[idx_start], x_scan[idx_end]

    num_slices = 30
    slice_edges = np.linspace(x_start, x_end, num_slices + 1)
    b = (x_end - x_start) / num_slices
    phi_rad = np.radians(phi)

    slices = []
    w_x = np.linspace(40, 130, 100)
    w_y = [get_phreatic_y(x) for x in w_x]

    for i in range(num_slices):
        x_mid = (slice_edges[i] + slice_edges[i + 1]) / 2
        y_top = np.interp(x_mid, dx, dy)
        y_bot = yc - np.sqrt(R**2 - (x_mid - xc) ** 2)
        h_slice = max(0, y_top - y_bot)

        y_center = y_bot + (h_slice / 2)
        hi = yc - y_center

        y_water = get_phreatic_y(x_mid)
        h_water = y_water - y_bot
        u_slice = h_water * gamma_w if h_water > 0 else 0

        W = h_slice * b * gamma
        alpha_rad = np.arcsin((x_mid - xc) / R)

        slices.append(
            {
                "W": W,
                "alpha_rad": alpha_rad,
                "b": b,
                "u": u_slice,
                "x_mid": x_mid,
                "h": h_slice,
                "y_bot": y_bot,
                "hi": hi,
            }
        )

    fs = 1.2
    convergence_history = []
    for i in range(25):
        convergence_history.append(fs)
        num, den = 0, 0
        for s in slices:
            a_rad = s["alpha_rad"]

            static_moment = s["W"] * np.sin(a_rad)
            seismic_moment = abs(kh * s["W"] * s["hi"] / R)
            den += static_moment + seismic_moment

            m_alpha = np.cos(a_rad) + (np.sin(a_rad) * np.tan(phi_rad) / fs)
            if m_alpha < 0.1:
                m_alpha = 0.1

            effective_weight = s["W"] - (s["u"] * s["b"])
            resisting = (
                c * s["b"] + max(0, effective_weight) * np.tan(phi_rad)
            ) / m_alpha
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
# 3. DIRECT SCRIPT EXECUTION TEST
# =====================================================================
if __name__ == "__main__":
    node_coords, elements, h_fem = run_post_rain_seepage()
    triangulation = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])

    fig, ax = plt.subplots(figsize=(10, 5))
    cf = ax.tricontourf(triangulation, h_fem, levels=20, cmap="viridis")

    for elem in elements:
        elem_nodes = elem + [elem[0]]
        ax.plot(
            node_coords[elem_nodes, 0],
            node_coords[elem_nodes, 1],
            "k-",
            linewidth=0.3,
            alpha=0.3,
        )

    ax.set_title("Post-Rain Unconfined Seepage Solution (run_post_rain_seepage)")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Elevation [m]")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, label="Hydraulic Head h [m]")

    plt.tight_layout()
    plt.show()
