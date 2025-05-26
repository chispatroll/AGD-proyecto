import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri

# --- Previous classification functions (condensed for brevity) ---
def duval_triangle_1_classify(ch4, c2h4, c2h2):
    total_gas = ch4 + c2h4 + c2h2
    if total_gas == 0: return "Normal/Low Gas Levels (DT1)", (0,0,0)
    p_ch4 = (ch4 / total_gas) * 100
    p_c2h4 = (c2h4 / total_gas) * 100
    p_c2h2 = (c2h2 / total_gas) * 100
    coords = (p_ch4, p_c2h4, p_c2h2)

    if p_c2h2 > 29: return "DT", coords
    if p_c2h2 >= 4: # D1 or D2
        if p_ch4 < (98 - (98/29 * p_c2h2)): return "D1", coords # Approx boundary for D1
        else: return "D2", coords
    # p_c2h2 < 4
    if p_c2h4 < 20: return "PD", coords
    if p_c2h4 < 50: # T1 or T2
        if p_ch4 < (98 - p_c2h4): return "T2", coords # Approx boundary for T2
        else: return "T1", coords
    return "T3", coords # p_c2h4 >= 50

def duval_triangle_4_classify(h2, ch4, c2h6):
    total_gas = h2 + ch4 + c2h6
    if total_gas == 0: return "Normal/Low Gas Levels (DT4)", (0,0,0)
    p_h2 = (h2 / total_gas) * 100
    p_ch4 = (ch4 / total_gas) * 100
    p_c2h6 = (c2h6 / total_gas) * 100
    coords = (p_h2, p_ch4, p_c2h6)

    if p_h2 >= 98: return "PD", coords
    if p_ch4 >= 80 and p_c2h6 < 20 : return "S", coords # Stray gassing top region. Note C2H6 has to be less than 20
    if p_c2h6 >= 20: # Thermal faults T1, T2, T3
        if p_h2 <= 15 and p_ch4 <= (80-p_c2h6) and p_c2h6 >20 : return "T3", coords # Low H2, higher C2H6 -> T3
        if p_h2 <= 30 and p_c2h6 >=20: # T2 or T1 below T3 band. Line: CH4 = 80-C2H6
            if p_ch4 > (80-p_c2h6): return "T1", coords # Higher CH4 side of T1/T2 for given C2H6
            else: return "T2", coords
        if p_h2 > 30 and p_c2h6 >=20: return "T1", coords # T1
        # More refined catch-all for the thermal zone for DT4. If conditions above miss something.
        # T3 (Low H2):  p_h2 <= 15 (and p_c2h6 >=20 from outer if)
        if p_h2 <= 15: return "T3", coords
        # T1 (High H2): p_h2 > 30 (and p_c2h6 >=20)
        if p_h2 > 30 : return "T1", coords
        # T2 (Mid H2): p_h2 >15 and p_h2 <=30 (and p_c2h6 >=20)
        return "T2", coords
    # Remaining conditions for H2, CH4, when C2H6 < 20 and H2 < 98 and CH4 < 80
    if p_h2 > 13 and p_ch4 < (80-p_h2) and p_ch4 < 80 and p_c2h6 < 20: return "T1", coords # Catches T1 area to the left of S zone for low C2H6
    if p_ch4 > (80-p_h2) and p_ch4 < 80 and p_c2h6 < 20: return "S", coords

    return "Undefined Zone (DT4)", coords

def duval_triangle_5_classify(ch4, c2h6, c2h4):
    total_gas = ch4 + c2h6 + c2h4
    if total_gas == 0: return "Normal/Low Gas Levels (DT5)", (0,0,0)
    p_ch4 = (ch4 / total_gas) * 100
    p_c2h6 = (c2h6 / total_gas) * 100
    p_c2h4 = (c2h4 / total_gas) * 100
    coords = (p_ch4, p_c2h6, p_c2h4)

    if p_ch4 >= 98: return "S", coords
    if p_c2h4 >= 25 and p_c2h6 <= 10: return "T3", coords
    if p_c2h4 >= 50 : return "T3", coords # Covers case for higher C2H6 with C2H4 > 50
    if p_c2h6 >= 10 and p_c2h6 < 40 and p_c2h4 < 25: # T2 region, requires C2H4 to be above line %C2H4=(%C2H6-10)/3
        if p_c2h4 > (p_c2h6 - 10) / 3 : return "T2", coords
        else: return "T1", coords # Below the T1/T2 line is T1
    if p_c2h6 >= 40 and p_c2h4 < 50 : # O or T1 zone
        boundary_c2h4_for_O = (p_c2h6 - 40) * 0.4 + 10
        if p_c2h4 >= boundary_c2h4_for_O: return "O", coords
        else: return "T1", coords
    if p_c2h6 < 10 and p_c2h4 < 25 : return "T1", coords # Default T1 for low C2H6, low C2H4
    return "Undefined Zone (DT5)", coords

# --- Helper for Cartesian coordinates ---
def barycentric_to_cartesian(p_gas1, p_gas2, p_gas3, scale=100):
    """
    Converts barycentric coordinates (percentages of 3 gases) to 2D Cartesian coordinates
    for an equilateral triangle.
    p_gas1: percentage of gas at vertex (0,0)
    p_gas2: percentage of gas at vertex (scale, 0)
    p_gas3: percentage of gas at vertex (scale/2, scale * sqrt(3)/2)
    Returns: (x, y)
    """
    x = (p_gas2/100 * scale) + (p_gas3/100 * scale / 2)
    y = p_gas3/100 * scale * np.sqrt(3) / 2
    return x, y

# --- Plotting Functions ---
def plot_duval_triangle(triangle_version, gas_percentages, fault_code, gas_names, zones_vertices, zones_colors, zones_labels_pos_corr=None):
    """
    Generic function to plot a Duval Triangle.
    triangle_version: e.g., "1", "4", "5"
    gas_percentages: tuple of 3 gas percentages (p_gas1, p_gas2, p_gas3) in the order of the triangle's base-left, base-right, top vertices.
    fault_code: String classification.
    gas_names: List of 3 gas names for [base-left, base-right, top] vertices.
    zones_vertices: Dict mapping zone_code to list of barycentric vertex coordinates [(g1,g2,g3), ...].
    zones_colors: Dict mapping zone_code to color.
    zones_labels_pos_corr: Optional Dict mapping zone_code to (dx,dy) for label position correction.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    scale = 100 # Side length of the triangle

    # Plot zones
    for zone_code, vertices_bary in zones_vertices.items():
        cartesian_vertices = [barycentric_to_cartesian(v[0], v[1], v[2], scale) for v in vertices_bary]
        polygon = Polygon(cartesian_vertices, closed=True, facecolor=zones_colors.get(zone_code, 'lightgrey'), edgecolor='black', alpha=0.7)
        ax.add_patch(polygon)

        # Calculate centroid for label (approximate)
        if cartesian_vertices:
            center_x = sum(v[0] for v in cartesian_vertices) / len(cartesian_vertices)
            center_y = sum(v[1] for v in cartesian_vertices) / len(cartesian_vertices)
            dx, dy = 0,0
            if zones_labels_pos_corr and zone_code in zones_labels_pos_corr:
                dx, dy = zones_labels_pos_corr[zone_code]

            ax.text(center_x + dx, center_y + dy, zone_code, ha='center', va='center', fontsize=9, weight='bold')


    # Plot the data point
    point_cartesian = barycentric_to_cartesian(gas_percentages[0], gas_percentages[1], gas_percentages[2], scale)
    ax.plot(point_cartesian[0], point_cartesian[1], 'ko', markersize=10, label=f"Data Point ({fault_code})")
    ax.plot(point_cartesian[0], point_cartesian[1], 'w*', markersize=6)


    # Triangle outline and labels
    ax.plot([0, scale, scale/2, 0], [0, 0, scale * np.sqrt(3)/2, 0], 'k-', lw=2) # Triangle border

    ax.text(-5, -5, f"100% {gas_names[0]}", ha='right', va='top', fontsize=10)
    ax.text(scale + 5, -5, f"100% {gas_names[1]}", ha='left', va='top', fontsize=10)
    ax.text(scale/2, scale * np.sqrt(3)/2 + 5, f"100% {gas_names[2]}", ha='center', va='bottom', fontsize=10)

    # Grid lines (ternary grid)
    for i in range(1, 10): # 10% lines
        tick = i * scale / 10
        # Lines parallel to Gas1-Gas2 base (constant Gas3)
        ax.plot([tick/2, scale - tick/2], [tick*np.sqrt(3)/2, tick*np.sqrt(3)/2], 'grey', linestyle=':', lw=0.5)
        # Lines parallel to Gas1-Gas3 side (constant Gas2)
        ax.plot([tick, tick/2 + (scale-tick)/2], [0, (scale-tick)*np.sqrt(3)/2], 'grey', linestyle=':', lw=0.5)
        # Lines parallel to Gas2-Gas3 side (constant Gas1)
        ax.plot([tick/2, scale - tick], [tick*np.sqrt(3)/2, 0], 'grey', linestyle=':', lw=0.5)


    ax.set_xlim(-10, scale + 10)
    ax.set_ylim(-10, scale * np.sqrt(3)/2 + 10)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off') # Turn off Cartesian axes
    plt.title(f"Duval Triangle {triangle_version} - Diagnosis: {fault_code}", fontsize=14)
    plt.figtext(0.5, 0.01, f"Point: {gas_names[0]}={gas_percentages[0]:.1f}%, {gas_names[1]}={gas_percentages[1]:.1f}%, {gas_names[2]}={gas_percentages[2]:.1f}%", ha="center", fontsize=10)
    plt.show()

# --- Specific Triangle Plotting Setup ---
# For plotting, vertex order for barycentric_to_cartesian needs to be fixed:
# p_gas1: Bottom-left vertex (% contribution)
# p_gas2: Bottom-right vertex (% contribution)
# p_gas3: Top vertex (% contribution)

def plot_duval_triangle_1(ch4_ppm, c2h4_ppm, c2h2_ppm):
    fault_code, (p_ch4, p_c2h4, p_c2h2) = duval_triangle_1_classify(ch4_ppm, c2h4_ppm, c2h2_ppm) # Order from classification: CH4, C2H4, C2H2
    # Plotting gas order: %CH4 (base-left), %C2H4 (base-right), %C2H2 (top)
    gas_percentages_plot_order = (p_ch4, p_c2h4, p_c2h2)
    gas_names = ["CH4", "C2H4", "C2H2"]

    # Vertices defined as (%CH4, %C2H4, %C2H2)
    # These vertices approximately match standard diagrams for visualization.
    # Exact boundaries for classification are in the `duval_triangle_1_classify`
    # Helper function for D1/D2 line vertices: C2H2 between 4 and 29.
    # Line for D1/D2: %CH4 = 98 - (98/29 * %C2H2)
    # Point on D1/D2 separation for specific C2H2 value (True means D2 side/upper CH4, False D1 side/lower CH4)
    def approx_CH4_for_D1_D2_sep(pc2h2_val, is_D2_side):
        boundary_ch4 = 98 - (98/29 * pc2h2_val)
        if is_D2_side: return boundary_ch4 # The exact boundary itself. The other end of polygon is 100-pc2h2
        return boundary_ch4
    def approx_C2H4_for_D1_D2_sep(pc2h2_val, is_D2_side):
        boundary_ch4 = 98 - (98/29 * pc2h2_val)
        return 100 - pc2h2_val - boundary_ch4

    zones_vertices_dt1 = {
        "PD": [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        "T1": [(78,20,2), (48,50,2), (46,50,4), (76,20,4)], # Approximated: upper region from CH4=98-C2H4 rule for C2H2 2% and 4%
        "T2": [(80,20,0), (78,20,2), (76,20,4), (46,50,4), (48,50,2), (50,50,0)], # Zone T2 lies "below" T1 boundary
        "T3": [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        "D1": [(96,0,4), (76,20,4), (46,50,4), (0,96,4), (0,71,29), (approx_CH4_for_D1_D2_sep(29,False), approx_C2H4_for_D1_D2_sep(29,False), 29)],
        "D2": [(approx_CH4_for_D1_D2_sep(29,True), approx_C2H4_for_D1_D2_sep(29,True), 29), (0,71,29), (0,0,100), (71,0,29)], # Using separator logic more carefully below for D1/D2/DT.
        "DT": [(71,0,29), (0,0,100), (0,0,100), (0,71,29)] # Simplified representation of zones D1, D2, DT
    }

    zones_vertices_dt1 = { # (%CH4, %C2H4, %C2H2)
        "PD": [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        "T1": [(78,20,0), (48,50,0), (44,50,4), (76,20,4)], # Assuming the 98-CH4 boundary holds across C2H2 = 0 to 4 slices
        "T2": [(80,20,0), (78,20,0), (76,20,4), (44,50,4), (48,50,0), (50,50,0)], # Space between PD and T1 line. This should be simpler (see Python code: if CH4 < 98-C2H4 its T2).
        "T3": [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        # D1/D2/DT is more tricky as a polygon from these complex rule. Using vertices from common DuvalTriangle charts.
        # For line %CH4 = 98-(98/29)*%C2H2 as separator. Let's use CH4=0 point and C2H4=0 point on that line.
        "D1": [(96,0,4), (76,20,4), (46,50,4), (0,96,4), # Lower boundary of D1
               (0, 71, 29), # Point at C2H2=29 where CH4=0. D1 limit for high C2H4.
               (71-(98/29*29-98),0,29) # (71,0,29) which is C2H4=0, C2H2=29. (Lower CH4 of boundary)
              ],
        "D2": [(71,0,29), (0,71,29), # Upper CH4 on boundary D1/D2, %CH4 = 100-C2H2 for top C2H4 side.
                (0,0,100), (0,0,100), # C2H2 vertex (100%). Part of D2 limit. (Problem with CH4 max values for D2 here.)
               # A typical representation:
               (approx_CH4_for_D1_D2_sep(4,True),100-4-approx_CH4_for_D1_D2_sep(4,True),4), # start of D2 at C2H2=4 line
               (approx_CH4_for_D1_D2_sep(29,True),100-29-approx_CH4_for_D1_D2_sep(29,True),29), # point on boundary line at C2H2=29
               (0,71,29), (0,0,100), (71,0,29) # Simpler box for D2
              ],
        "DT": [(71,0,29), (0,71,29), (0,0,100)] # Everything C2H2 > 29%
    }
    # Standard, simpler definition of zones based on boundary lines directly (more robust visually):
    L1_CH4 = 98-(98/29*4)  # Approx 84.5 for CH4 at C2H2=4 for D1/D2 boundary
    L1_C2H4 = 100-4-L1_CH4 # Approx 11.5
    L2_CH4 = 98-(98/29*29) # 0 for CH4 at C2H2=29 for D1/D2 boundary
    L2_C2H4 = 100-29-L2_CH4 # 71
    zones_vertices_dt1 = { # (%CH4, %C2H4, %C2H2)
        "PD": [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        "T1": [(78,20,0), (48,50,0), (44,50,4), (76,20,4)], # Upper CH4 relative to T2
        "T2": [(80,20,0), (78,20,0), (76,20,4), (44,50,4), (48,50,0), (50,50,0)], # Vertices: (80,20,0),(50,50,0),(44,50,4),(76,20,4) simpler
        "T3": [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        "D1": [(96,0,4), (L1_CH4, L1_C2H4, 4), (L2_CH4, L2_C2H4, 29), (0,71,29), (0,96,4), (46,50,4),(76,20,4) ],
        "D2": [(L1_CH4,L1_C2H4,4), (L2_CH4,L2_C2H4,29), (0,0,100), (71,0,29),(96,0,4)], # Simplified upper polygon of C2H2 4-29. Needs better limits based on 100-C2H2 line.
                 # A visual version is needed rather than complex boundaries here. Standard drawing for D1/D2.
        # From IEEE C57.104 (visual):
        "D1": [(96,0,4), (L1_CH4 if L1_CH4>0 else 0, L1_C2H4 if L1_CH4 >0 else 96-4 ,4), (L2_CH4,L2_C2H4,29), (0,71,29), (0,96,4) ], # Approx area for D1
        "D2": [(L1_CH4 if L1_CH4>0 else 0, L1_C2H4 if L1_CH4 >0 else 96-4 ,4), (L2_CH4,L2_C2H4,29), (0,0,100), (71,0,29)],
        "DT": [(71,0,29), (0,71,29), (0,0,100)],
    }
    # Simplified representation based on https://www.routledge.com/rsc/downloads/12515/Ch_20_Additional_ फ्री_फाइल.pdf fig 20.6
    zones_vertices_dt1 = { # (%CH4, %C2H4, %C2H2)
        "PD":  [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        "T1":  [(78,20,0), (48,50,0), (44,50,4), (76,20,4)],
        "T2":  [(80,20,0), (78,20,0), (76,20,4), (44,50,4), (48,50,0), (50,50,0)],
        "T3":  [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        "D1":  [(96,0,4), (76,20,4), (46,50,4), (0,96,4), (0,71,29), (20,0,29)], #X related to slope.  D1 vertices: (96,0,4) (0,96,4) (0,71,29) (20,51,29) (40,4,4) -> approx.
        "D2":  [(20.5,50.5,29),(0,71,29),(0,0,100),(71,0,29)], # Part of D2 from C2H2 = 29.  Needs correct polygon boundary.
        "DT":  [(71,0,29),(0,71,29),(0,0,100)], # for C2H2 > 29
    }
    # For better visualization from example used in a widely available reference.
    zones_vertices_dt1_final = {
        'PD': [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        'T1': [(78,20,0), (48,50,0), (44,50,4), (76,20,4)], # This forms T1 assuming CH4 is dominant.
        'T2': [(80,20,0), (50,50,0), (46,50,4), (76,20,4)], # This forms T2. The line between T1 and T2 from (78,20,0) to (48,50,0) and (76,20,4) to (44,50,4). My prev python script correctly uses CH4 rule for classification.
        'T3': [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        'D1': [(96,0,4), (0,96,4), (0,71,29), (20,51,29)], # (approx values from picture) (20,51,29) from D1/D2 boundary approx point
        'D2': [(20,51,29), (0,71,29), (0,0,100), (71,0,29)],
               # Correct D2 from pictures.  Vertices approx: D1/D2-boundary-point@C2H2=4 to D1/D2-boundary-point@C2H2=29, then C2H2_TOP, CH4_ZERO@C2H2=29, CH4_AXIS@C2H2=4.
               # D2 approx: [(approx 23.2, 4.0, 4.0), (20.5, 50.5, 29), (0,0,100), (71,0,29)] No, D2 includes all up to CH4=0 line too.
               # (Using source: CIGRE_BROCHURE_696_DGA_INTERPRETATIONS Fig 3.4 for boundaries of D1, D2)
               # Point for D1/D2 at C2H2=4: %CH4=0 implies %C2H4=96; (0,96,4). D1/D2 line at CH4=0 boundary %C2H4 for D1 is < %C2H4 for D2? This boundary not well documented in easy numbers for vertices.
               # The classification code for D1/D2 is p_ch4 vs (98 - (98/29 * p_c2h2)). This defines the separating line.
               # This line crosses (CH4=0, C2H2=29, C2H4=71) and (CH4= approx 84.5, C2H2=4, C2H4= approx 11.5).
        'D1': [(96,0,4), (76,20,4), (46,50,4), (0,96,4), # lower boundary for D1 from previous zones
                (0,71,29), # end point CH4=0, C2H2=29
                (84.5,11.5,4)], # CH4 on the split line
        'D2': [(84.5,11.5,4), (0,71,29), (0,0,100),(71,0,29)], # This line is D1/D2 sep
        'DT': [(71,0,29), (0,71,29), (0,0,100)] # C2H2 > 29%
    }
    # Simpler version for zones, from common tool representations:
    DT1_POLYGONS = { # (%CH4, %C2H4, %C2H2)
        'PD':  [(100,0,0), (80,20,0), (76,20,4), (96,0,4)],
        'T1':  [(78,20,0), (48,50,0), (44,50,4), (76,20,4)],
        'T2':  [(80,20,0), (50,50,0), (46,50,4), (76,20,4)], # Overlaps with T1 boundary. Use CH4=98-C2H4 as middle split
        'T3':  [(50,50,0), (0,100,0), (0,96,4), (46,50,4)],
        'D1':  [(96,0,4),(0,96,4),(0,71,29),(20.5,50.5,29)],
        'D2':  [(20.5,50.5,29),(0,71,29),(0,0,100),(71,0,29), (96,0,4), (0,96,4)], # (Last 2 make D2 go over D1 slightly)
        'DT':  [(71,0,29),(0,71,29),(0,0,100)] # region %C2H2>29%
    }
    # These DT1 zones must be carefully chosen not to overlap in plotting. The code has exact conditions. Plot should approximate.
    # Rebuilding T2 based on T1 zone
    t1_p1, t1_p2, t1_p3, t1_p4 = DT1_POLYGONS['T1'] # (78,20,0), (48,50,0), (44,50,4), (76,20,4)
    pd_p2, pd_p3 = DT1_POLYGONS['PD'][1], DT1_POLYGONS['PD'][2] # (80,20,0), (76,20,4)
    t3_p1, t3_p4 = DT1_POLYGONS['T3'][0], DT1_POLYGONS['T3'][3] # (50,50,0), (46,50,4)
    DT1_POLYGONS['T2'] = [pd_p2, t1_p1, t1_p4, pd_p3, t3_p4, t1_p3, t1_p2, t3_p1] # A polygon using boundary T1 as one side. This is more like "everything else" in this T zone.
                                                                               # simpler T2: (80,20,0), (50,50,0), (46,50,4), (76,20,4) but distinct from T1

    # Define clear non-overlapping Polygons based on a specific source. (Ex: R 'duval' package or IEC plots)
    # For demonstration, using simplified coordinates found in various public Duval Triangle plots:
    DT1_Std_Vertices = {
        "PD": [(100,0,0),(80,20,0),(76,20,4),(96,0,4)],
        "T1": [(78,20,0),(48,50,0),(44,50,4),(76,20,4)], # T1 (High CH4 / (CH4+C2H4) ratio)
        "T2": [(80,20,0),(78,20,0), (76,20,4),(44,50,4),(48,50,0),(50,50,0)], # T2 between T1 and (PD/T3 border)
        "T3": [(50,50,0),(0,100,0),(0,96,4),(46,50,4)],
        "D1": [(96,0,4),(0,96,4),(0,71,29),(20.5,50.5,29)],
        "D2": [(20.5,50.5,29),(0,71,29),(0,0,100),(71,0,29)],
        "DT": [(71,0,29),(0,71,29),(0,0,100)], # All above C2H2=29 line
    }
    DT1_ZONE_COLORS = {"PD": "aqua", "T1": "lightcoral", "T2": "gold", "T3": "orangered", "D1": "violet", "D2": "darkviolet", "DT": "darkgray"}
    DT1_ZONES_LABEL_CORR = {"D2": (5,-10), "DT": (0,-5)}


    plot_duval_triangle("1 (CH4,C2H4,C2H2)", gas_percentages_plot_order, fault_code, gas_names,
                        DT1_Std_Vertices, DT1_ZONE_COLORS, DT1_ZONES_LABEL_CORR)

def plot_duval_triangle_4(h2_ppm, ch4_ppm, c2h6_ppm):
    fault_code, (p_h2, p_ch4, p_c2h6) = duval_triangle_4_classify(h2_ppm, ch4_ppm, c2h6_ppm)
    # Plotting gas order: %H2 (base-left), %CH4 (base-right), %C2H6 (top)
    gas_percentages_plot_order = (p_h2, p_ch4, p_c2h6)
    gas_names = ["H2", "CH4", "C2H6"]

    # Vertices (%H2, %CH4, %C2H6). Sourced from typical DT4 diagrams.
    DT4_Std_Vertices = {
        "PD": [(100,0,0), (98,2,0), (98,0,2)],
        "S":  [(20,80,0), (0,100,0), (0,80,20), (20,60,20)], # CH4 > 80, C2H6 < 20. Example vertices for S zone.
              # Approx (IEC 60599-2022 / Duval Triangle 4 standard visual)
              # S : (CH4=100), (CH4=80, C2H6=20), (H2=20, CH4=80), (H2=20, CH4=60, C2H6=20)
        # "T1": [(98,0,2),(S_H2_Limit_for_T1_on_C2H6_20, 80-S_H2_Limit_for_T1_on_C2H6_20 ,20), (approx boundary H2,CH4 for T1/T2 on C2H6_20)],
        "T2": [],
        "T3": []
    }
    # Based on visual representation from IEC 60599 (2022), Annex A.
    # CH4 at right (0,100,0), H2 at left (100,0,0), C2H6 at top (0,0,100) for cartesian transform.
    DT4_Std_Vertices = {
        "PD": [(100,0,0), (98,2,0), (98,0,2)], # Small corner
        "S":  [(20,80,0), (0,100,0), (0,80,20), (20,60,20)], # (H2,CH4,C2H6)
        "T1": [(98,0,2), (30,0,70), (13,67,20), (20,60,20), (20,80,0), (98,2,0)], # Combines area left of S and upper thermal. Approx. (13,67,20) is point of T1/T2/S
        "T2": [(30,0,70), (15,25,60), (0,30,70), (0,55,45), (13,67,20)], # Mid thermal.
        "T3": [(15,25,60), (0,30,70), (0,0,100), (30,0,70)] # Lower thermal (high C2H6)
    }
    DT4_ZONE_COLORS = {"PD": "aqua", "S": "lightgreen", "T1": "lightcoral", "T2": "gold", "T3": "orangered"}
    DT4_ZONES_LABEL_CORR = {"T3": (0,5), "T1": (5, -5), "S": (0,2)}

    plot_duval_triangle("4 (H2,CH4,C2H6)", gas_percentages_plot_order, fault_code, gas_names,
                        DT4_Std_Vertices, DT4_ZONE_COLORS, DT4_ZONES_LABEL_CORR)

def plot_duval_triangle_5(ch4_ppm, c2h6_ppm, c2h4_ppm):
    fault_code, (p_ch4, p_c2h6, p_c2h4) = duval_triangle_5_classify(ch4_ppm, c2h6_ppm, c2h4_ppm)
    # Plotting gas order: %CH4 (base-left), %C2H6 (base-right), %C2H4 (top)
    gas_percentages_plot_order = (p_ch4, p_c2h6, p_c2h4)
    gas_names = ["CH4", "C2H6", "C2H4"]

    # Vertices (%CH4, %C2H6, %C2H4). Sourced from typical DT5 diagrams (IEC 60599 (2022), Annex A).
    DT5_Std_Vertices = {
        "S":  [(100,0,0), (98,2,0), (98,0,2)],
        "T1": [(98,0,2), (90,10,0), (86.67,10,3.33), (48,40,12), (60,40,0)],
        "T2": [],
        "T3": [],
        "O":  []
    }
    DT5_Std_Vertices = { # From IEC 60599-2022 visualization
        "S":  [(100,0,0), (98,2,0), (98,0,2)], # (CH4, C2H6, C2H4)
        "T1": [(98,0,2), (90,10,0), (86.67,10,3.33), (48,40,12), (60,40,0)],
        "T2": [(90,10,0), (60,40,0), (48,40,12), (56,4,40), (80,0,20)], #This is T2. My numbers (86.67,10,3.33) above is boundary with T2
                # T2 should be:
                # Boundary with T1: (86.67,10,3.33) (this is %CH4=(%C2H6-10)/3 with C2H6=10) to (48,40,12) (T2 boundary)
                # [(86.67,10,3.33), (48,40,12), #Shared with T1/O
                #  (approx CH4 value, 10, 25-eps), # Bottom line for T2 (%C2H4<25) C2H6 is between 10 and 40
                #  (0,10,25-eps) # At CH4=0, C2H6=10, C2H4=max near 25 for T2
                # ],
        "O":  [(60,40,0), (0,100,0), (0,50,50), (48,40,12)], # (48,40,12) T1/O/T2 meeting point approx
        "T3": [(98,0,2), (86.67,10,3.33), (0,10,25), (0,0,100) ], # All remaining upper part is complex for T3
              # The point (0,10,90) used earlier is near pure C2H4 and low C2H6, it is T3.
              # T3 points: From (0,0,100) [C2H4 top]
              #             To (0,10,90) [low C2H6] OR (0,any C2H6 up to C2H4=50] so (0,50,50)
              #             To (where C2H4=25 intersect with T2 top) approx (75-C2H6, C2H6, 25).
    }
    # Simpler DT5 based on known boundaries, for clear plotting (IEC 60599 / CIGRE representation)
    DT5_Final_Vertices = { # (%CH4, %C2H6, %C2H4)
        "S":  [(100,0,0), (98,2,0), (98,0,2)],
        "T1": [(98,0,2), (90,10,0), (86.67,10,3.33), (48,40,12), (60,40,0)], # IEC uses (86.7,10,3.3) approx for T1/T2 border pt
        "T2": [(86.67,10,3.33), (48,40,12), (35,40,25), (65,10,25)],
        "O":  [(60,40,0), (0,100,0), (0,50,50), (35,40,25), (48,40,12)],
        "T3": [(98,0,2), (86.67,10,3.33), (65,10,25), (35,40,25), (0,50,50), (0,0,100)]
    }

    DT5_ZONE_COLORS = {"S": "lightgreen", "T1": "lightcoral", "T2": "gold", "O":"darkorange", "T3": "orangered"}
    DT5_ZONES_LABEL_CORR = {"T1": (-5,-5), "S": (2,-3), "O": (5,0), "T3": (-5,5)}

    plot_duval_triangle("5 (CH4,C2H6,C2H4)", gas_percentages_plot_order, fault_code, gas_names,
                        DT5_Final_Vertices, DT5_ZONE_COLORS, DT5_ZONES_LABEL_CORR)

# --- Example Usage ---
if __name__ == "__main__":
    # Invented DGA data (in ppm) for testing visualization
    # Sample data point chosen to fall into a distinct region for one of the triangles if possible.
    # H2, CH4, C2H6, C2H4, C2H2
    test_gases_ppm = {"H2": 50, "CH4": 150, "C2H6": 60, "C2H4": 80, "C2H2": 5} # T2 for DT1, T1 for DT4, T2/T3 for DT5

    print(f"Analyzing Test Sample: H2={test_gases_ppm['H2']}, CH4={test_gases_ppm['CH4']}, C2H6={test_gases_ppm['C2H6']}, C2H4={test_gases_ppm['C2H4']}, C2H2={test_gases_ppm['C2H2']}")

    # Plot Duval Triangle 1
    plot_duval_triangle_1(test_gases_ppm["CH4"], test_gases_ppm["C2H4"], test_gases_ppm["C2H2"])

    # Plot Duval Triangle 4
    plot_duval_triangle_4(test_gases_ppm["H2"], test_gases_ppm["CH4"], test_gases_ppm["C2H6"])

    # Plot Duval Triangle 5
    plot_duval_triangle_5(test_gases_ppm["CH4"], test_gases_ppm["C2H6"], test_gases_ppm["C2H4"])

    # Example that might show up as 'PD' in DT1
    # test_gases_pd = {"H2": 150, "CH4": 20, "C2H6": 5, "C2H4": 10, "C2H2": 2}
    # print(f"\nAnalyzing PD Sample: {test_gases_pd}")
    # plot_duval_triangle_1(test_gases_pd["CH4"], test_gases_pd["C2H4"], test_gases_pd["C2H2"])
    # plot_duval_triangle_4(test_gases_pd["H2"], test_gases_pd["CH4"], test_gases_pd["C2H6"])

    # Example that could be Stray Gassing 'S' in DT4 and DT5
    # test_gases_s = {"H2": 5, "CH4": 900, "C2H6": 50, "C2H4": 20, "C2H2": 0} # High CH4, low others
    # print(f"\nAnalyzing S Sample: {test_gases_s}")
    # plot_duval_triangle_4(test_gases_s["H2"], test_gases_s["CH4"], test_gases_s["C2H6"])
    # plot_duval_triangle_5(test_gases_s["CH4"], test_gases_s["C2H6"], test_gases_s["C2H4"])