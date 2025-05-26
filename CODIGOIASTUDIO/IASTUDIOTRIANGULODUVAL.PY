# Duval Triangle 1 (DT1)
# Gases: CH4 (Methane), C2H4 (Ethylene), C2H2 (Acetylene)
# Fault Codes: PD, T1, T2, T3, D1, D2, DT
def duval_triangle_1(ch4, c2h4, c2h2):
    """
    Calculates fault type based on Duval Triangle 1.
    Inputs:
        ch4 (float): Methane concentration (ppm)
        c2h4 (float): Ethylene concentration (ppm)
        c2h2 (float): Acetylene concentration (ppm)
    Returns:
        str: Fault zone code (PD, T1, T2, T3, D1, D2, DT) or "Normal/Low Gas"
    """
    total_gas = ch4 + c2h4 + c2h2

    if total_gas == 0:
        return "Normal/Low Gas Levels (DT1)"

    # Calculate percentages
    p_ch4 = (ch4 / total_gas) * 100
    p_c2h4 = (c2h4 / total_gas) * 100
    p_c2h2 = (c2h2 / total_gas) * 100

    # Zone definitions for Duval Triangle 1
    # Note: Order of checks can be important due to overlapping boundary conditions in logic.
    # These are based on standard definitions.

    if p_c2h2 > 29: # DT Zone (Mixed thermal and electrical)
        return "DT"
    elif p_c2h2 >= 4 and p_c2h2 <= 29: # D1 or D2
        if p_ch4 < (98 - (98/29 * p_c2h2)): # Below the D1/D2 dividing line
            return "D1"
        else: # Above or on the D1/D2 dividing line
            return "D2"
    # p_c2h2 < 4: Potential for PD, T1, T2, T3
    elif p_c2h4 < 20: # PD Zone
        return "PD"
    elif p_c2h4 >= 20 and p_c2h4 < 50: # T1 or T2 Zone
        # Line separating T1 and T2 (approximately CH4 = 98 - C2H4)
        # Or more precisely, slope from (C2H4=20, CH4=78) to (C2H4=50, CH4=48)
        # Equation of line: %CH4 = 98 - %C2H4 (approx.)
        # Or more accurately: %CH4 = 98 - (%C2H4 * (98-48)/(50-20)) is wrong
        # The line connects (CH4=98, C2H2=0, C2H4=0) effectively to (CH4=48, C2H2=0, C2H4=50) for T1/T2 boundary
        # The CIGRE Brochure 696 uses a line with slope -1 for T1/T2:
        # %CH4 >= 98 - %C2H4 for T1
        # This seems to be an error in some popular depictions.
        # A common boundary for T1/T2 when C2H2 < 4 & 20 <= C2H4 < 50 is:
        # (CH4 coordinate of T1/T2 line): Y = 98 - (X * (98-48)/(50-0)) where X is C2H4
        # Simplified: Check if %CH4 is greater than boundary value for T2
        # A very common definition:
        if p_ch4 < (98 - p_c2h4): # T2 Zone (below line CH4 + C2H4 = 98)
            return "T2"
        else: # T1 Zone (above line CH4 + C2H4 = 98)
            return "T1"
    elif p_c2h4 >= 50: # T3 Zone
        return "T3"
    else:
        return "Undefined Zone (DT1)" # Should not be reached if logic is correct

# Duval Triangle 4 (DT4)
# Gases: H2 (Hydrogen), CH4 (Methane), C2H6 (Ethane)
# Fault Codes: PD, T1, T2, T3, S (Stray Gassing)
def duval_triangle_4(h2, ch4, c2h6):
    """
    Calculates fault type based on Duval Triangle 4.
    Inputs:
        h2 (float): Hydrogen concentration (ppm)
        ch4 (float): Methane concentration (ppm)
        c2h6 (float): Ethane concentration (ppm)
    Returns:
        str: Fault zone code (PD, T1, T2, T3, S) or "Normal/Low Gas"
    """
    total_gas = h2 + ch4 + c2h6

    if total_gas == 0:
        return "Normal/Low Gas Levels (DT4)"

    p_h2 = (h2 / total_gas) * 100
    p_ch4 = (ch4 / total_gas) * 100
    p_c2h6 = (c2h6 / total_gas) * 100

    # Zone definitions for Duval Triangle 4
    if p_h2 >= 98: # PD Zone
        return "PD"
    elif p_ch4 >= 80 and p_c2h6 < 20: # S Zone (Stray Gassing)
         return "S"
    elif p_c2h6 >= 20: # Thermal faults T1, T2, T3 (lower part of triangle)
        if p_h2 <= 15 and p_ch4 <= 30: # T3 Zone (bottom right)
             return "T3"
        elif p_h2 <= 30 and p_c2h6 >= 20 and p_c2h6 <=70 : # T2 zone (middle)
             # Boundary between T2 and T1: %CH4 = 80 - %C2H6
             if p_ch4 > (80 - p_c2h6) and p_h2 <=30: # This checks CH4 is 'higher' than boundary
                 return "T1"
             else:
                 return "T2"
        elif p_h2 > 30 and p_c2h6 >=20: # T1 zone (left part of thermal)
            return "T1"
        else: # This section refines T1/T2/T3 logic based on typical shapes
            # T3: High C2H6, low H2, low-mid CH4
            if p_c2h6 >= 70 and p_h2 <= 15:
                return "T3"
            # T1: Higher H2, can have varied C2H6
            elif p_h2 > 13 and p_ch4 < 80 - p_c2h6: # Simplified T1 condition for clarity
                return "T1"
            # T2: Typically mid-range for H2 and C2H6
            else:
                # More specific T2 (often a band)
                # T2 is %H2 <= 30, %C2H6 >= 20, %CH4 < (80-%C2H6) (approx)
                if p_h2 <= 30 and p_ch4 < (80 - p_c2h6) :
                    return "T2"
                elif p_h2 <= 15 and p_ch4 >= 30: # This can be T3 or T2
                    if p_c2h6 >= 70: return "T3"
                    else: return "T2" # Default for mid C2H6
                else: # Default to T1 if conditions not met
                    return "T1" # Fallback for this branch if other T2/T3 not hit.
    # This area usually S or T1 based on H2/CH4
    elif p_ch4 >= 80-p_h2 and p_ch4 < 80: # S region
        return "S"
    elif p_h2 < 98 and p_h2 > 30 and p_ch4 < (98-p_h2): # T1 region to left of PD
        return "T1"

    return "Undefined Zone (DT4)" # Fallback

# Duval Triangle 5 (DT5)
# Gases: CH4 (Methane), C2H6 (Ethane), C2H4 (Ethylene)
# Fault Codes: T1, T2, T3, S (Stray Gassing), O (Overheating of Oil)
def duval_triangle_5(ch4, c2h6, c2h4):
    """
    Calculates fault type based on Duval Triangle 5.
    Inputs:
        ch4 (float): Methane concentration (ppm)
        c2h6 (float): Ethane concentration (ppm)
        c2h4 (float): Ethylene concentration (ppm)
    Returns:
        str: Fault zone code (T1, T2, T3, S, O) or "Normal/Low Gas"
    """
    total_gas = ch4 + c2h6 + c2h4

    if total_gas == 0:
        return "Normal/Low Gas Levels (DT5)"

    p_ch4 = (ch4 / total_gas) * 100
    p_c2h6 = (c2h6 / total_gas) * 100
    p_c2h4 = (c2h4 / total_gas) * 100

    # Zone definitions for Duval Triangle 5
    if p_ch4 >= 98: # S Zone (Stray Gassing)
        return "S"
    elif p_c2h4 >= 25 and p_c2h6 <= 10 : # T3 Zone (high C2H4, low C2H6)
        return "T3"
    elif p_c2h4 >= 50 : # Also T3 (very high C2H4 covers cases where C2H6 > 10)
        return "T3"
    elif p_c2h6 >= 10 and p_c2h6 < 40 and p_c2h4 < 25: # T2 Zone
        # Boundary T1/T2 is roughly %C2H4 = (%C2H6 - 10) / 3
        # If %C2H4 < (%C2H6 - 10) / 3, it's T1, otherwise T2 for this C2H6 range
        if p_c2h4 > (p_c2h6 - 10) / 3 :
             return "T2"
        else:
             return "T1"
    elif p_c2h6 >= 40 and p_c2h4 < 50 : # O or T1 Zone (Overheating or Low Temp Thermal)
        # Boundary T1/O is %C2H4 = (%C2H6 - 40) * 0.4 + 10
        # If %C2H4 < boundary, it's T1, otherwise O
        boundary_c2h4_for_O = (p_c2h6 - 40) * 0.4 + 10
        if p_c2h4 >= boundary_c2h4_for_O:
            return "O"
        else:
            return "T1"
    elif p_c2h6 < 10 and p_c2h4 < 25: # T1 Zone (low C2H6, low C2H4)
        return "T1"

    return "Undefined Zone (DT5)" # Fallback

# --- Main execution example ---
if __name__ == "__main__":
    # Example DGA data sets (in ppm)
    # Format: [H2, CH4, C2H6, C2H4, C2H2, CO, CO2] - CO, CO2 not used by these triangles
    # Source of examples: Some are typical, some are made to hit specific zones.

    sample_data = [
        {"name": "Sample 1 (PD type)", "gases": {"H2": 150, "CH4": 20, "C2H6": 5, "C2H4": 10, "C2H2": 2}}, # Expect PD from DT1/DT4
        {"name": "Sample 2 (T1 type)", "gases": {"H2": 60, "CH4": 100, "C2H6": 50, "C2H4": 30, "C2H2": 1}}, # Expect T1
        {"name": "Sample 3 (T2 type)", "gases": {"H2": 50, "CH4": 60, "C2H6": 80, "C2H4": 80, "C2H2": 3}}, # Expect T2
        {"name": "Sample 4 (T3 type)", "gases": {"H2": 10, "CH4": 10, "C2H6": 20, "C2H4": 150, "C2H2": 3}}, # Expect T3
        {"name": "Sample 5 (D1 type)", "gases": {"H2": 30, "CH4": 20, "C2H6": 10, "C2H4": 70, "C2H2": 10}}, # Expect D1
        {"name": "Sample 6 (D2 type)", "gases": {"H2": 20, "CH4": 70, "C2H6": 5, "C2H4": 80, "C2H2": 15}},  # Expect D2
        {"name": "Sample 7 (DT mix)", "gases": {"H2": 100, "CH4": 100, "C2H6": 50, "C2H4": 100, "C2H2": 50}}, # Expect DT
        {"name": "Sample 8 (Stray Gassing DT4/5)", "gases": {"H2": 5, "CH4": 90, "C2H6": 5, "C2H4": 2, "C2H2": 0}}, # Expect S
        {"name": "Sample 9 (Overheating Oil DT5)", "gases": {"H2": 20, "CH4": 30, "C2H6": 100, "C2H4": 30, "C2H2": 1}}, # Expect O
        {"name": "Sample 10 (Low Gas)", "gases": {"H2": 0, "CH4": 0, "C2H6": 0, "C2H4": 0, "C2H2": 0}},
        {"name": "Sample 11 (DT1 - T1 boundary)", "gases": {"H2": 0, "CH4": 60, "C2H6": 0, "C2H4": 39, "C2H2": 1}}, # (60% CH4, 39% C2H4, 1% C2H2) -> T1
        {"name": "Sample 12 (DT1 - T2 boundary)", "gases": {"H2": 0, "CH4": 20, "C2H6": 0, "C2H4": 39, "C2H2": 1}}, # (20% CH4, 78% C2H4, 2% C2H2) -> T2
        {"name": "Sample 13 (DT4 - T3 near C2H6 axis)", "gases": {"H2": 5, "CH4": 10, "C2H6": 85, "C2H4": 0, "C2H2": 0}}, # Expect T3 from DT4
        {"name": "Sample 14 (DT4 - PD)", "gases": {"H2": 99, "CH4": 0.5, "C2H6": 0.5, "C2H4": 0, "C2H2": 0}}, # Expect PD from DT4
    ]

    print("--- Duval Triangle Classifications ---")
    for item in sample_data:
        g = item["gases"]
        print(f"\nAnalyzing: {item['name']}")

        # Duval Triangle 1
        dt1_fault = duval_triangle_1(g["CH4"], g["C2H4"], g["C2H2"])
        print(f"  Duval Triangle 1 (CH4, C2H4, C2H2): {dt1_fault}")

        # Duval Triangle 4
        dt4_fault = duval_triangle_4(g["H2"], g["CH4"], g["C2H6"])
        print(f"  Duval Triangle 4 (H2, CH4, C2H6):   {dt4_fault}")

        # Duval Triangle 5
        dt5_fault = duval_triangle_5(g["CH4"], g["C2H6"], g["C2H4"])
        print(f"  Duval Triangle 5 (CH4, C2H6, C2H4): {dt5_fault}")

    print("\n--- Fault Type Key ---")
    print("PD: Partial Discharges (Electrical)")
    print("D1: Low Energy Discharges (Electrical)")
    print("D2: High Energy Discharges (Electrical)")
    print("T1: Thermal Fault, T < 300 °C (Thermal)")
    print("T2: Thermal Fault, 300 °C < T < 700 °C (Thermal)")
    print("T3: Thermal Fault, T > 700 °C (Thermal)")
    print("DT: Mixture of Electrical and Thermal faults (DT1 only)")
    print("S:  Stray Gassing / Oil Contamination (DT4, DT5)")
    print("O:  Overheating of Oil (DT5 only)")
    print("Normal/Low Gas Levels: Gas levels too low for diagnosis or sum of relevant gases is zero.")
    print("Undefined Zone: Gas percentages fall outside defined zones (should be rare with correct logic).")