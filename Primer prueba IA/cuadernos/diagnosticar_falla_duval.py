import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib # Para cargar el modelo y el LabelEncoder
import os # Para verificar rutas de archivos
import warnings # Para manejar warnings específicos si es necesario

# --- Nombres de Archivos del Modelo y LabelEncoder ---
# Se asume que estos archivos están en el MISMO DIRECTORIO que este script.
MODEL_FILENAME = 'modelo_pentagono_rf.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder_pentagono.pkl'

# Nombres de las características con las que se entrenó el modelo
FEATURE_NAMES = ['x_centroid', 'y_centroid']

# --- 1. Funciones para el Cálculo del Centroide ---
SUMMIT_COORDS_100_SCALE = {
    "H2":   (0, 100.0), "C2H6": (-95.10565162951536, 30.90169943749474),
    "CH4":  (-58.77852522924731, -80.90169943749475), "C2H4": (58.77852522924732, -80.90169943749473),
    "C2H2": (95.10565162951535, 30.901699437494745)
}
GAS_ORDER_FOR_POLYGON = ["H2", "C2H6", "CH4", "C2H4", "C2H2"]

def calculate_duval_centroid_from_percentages(gas_percentages_dict):
    polygon_vertices = []
    for gas_name in GAS_ORDER_FOR_POLYGON:
        percentage = gas_percentages_dict.get(gas_name, 0.0)
        summit_x, summit_y = SUMMIT_COORDS_100_SCALE[gas_name]
        polygon_vertices.append((percentage * summit_x, percentage * summit_y))

    pv = np.array(polygon_vertices)
    x = pv[:, 0]
    y = pv[:, 1]

    A_duval_term = 0.0
    for i in range(len(pv)):
        j = (i + 1) % len(pv)
        A_duval_term += (x[i] * y[j] - x[j] * y[i])
    A_duval_calc = 0.5 * A_duval_term

    if abs(A_duval_calc) < 1e-9:
        if all(abs(val) < 1e-9 for val in gas_percentages_dict.values()):
            return 0.0, 0.0
        return np.mean(x), np.mean(y)

    cx_sum_term = 0.0
    cy_sum_term = 0.0
    for i in range(len(pv)):
        j = (i + 1) % len(pv)
        common_term = (x[i] * y[j] - x[j] * y[i])
        cx_sum_term += (x[i] + x[j]) * common_term
        cy_sum_term += (y[i] + y[j]) * common_term

    Cx = (1 / (6 * A_duval_calc)) * cx_sum_term
    Cy = (1 / (6 * A_duval_calc)) * cy_sum_term
    return Cx, Cy

def calculate_centroid_from_ppm(h2_ppm, ch4_ppm, c2h2_ppm, c2h4_ppm, c2h6_ppm):
    gases_ppm = {"H2": h2_ppm, "CH4": ch4_ppm, "C2H2": c2h2_ppm, "C2H4": c2h4_ppm, "C2H6": c2h6_ppm}
    total_ppm = sum(gases_ppm.values())
    if total_ppm == 0: return 0.0, 0.0
    gas_percentages = {name: ppm / total_ppm for name, ppm in gases_ppm.items()}
    return calculate_duval_centroid_from_percentages(gas_percentages)

# --- 2. Función Principal de Diagnóstico ---
def diagnose_dga_sample(h2_ppm, ch4_ppm, c2h2_ppm, c2h4_ppm, c2h6_ppm, model_file, encoder_file, features):
    """
    Carga el modelo, calcula el centroide y predice la falla.
    """
    # Verificar si los archivos del modelo existen
    if not os.path.exists(model_file):
        print(f"Error Crítico: Archivo del modelo NO ENCONTRADO en: {os.path.abspath(model_file)}")
        return "Error de Carga", (None, None)
    if not os.path.exists(encoder_file):
        print(f"Error Crítico: Archivo del LabelEncoder NO ENCONTRADO en: {os.path.abspath(encoder_file)}")
        return "Error de Carga", (None, None)

    # Cargar el modelo y el LabelEncoder
    try:
        ml_model = joblib.load(model_file)
        le = joblib.load(encoder_file)
        # print(f"Modelo ('{model_file}') y LabelEncoder ('{encoder_file}') cargados exitosamente.")
    except Exception as e:
        print(f"Ocurrió un error al cargar el modelo o el LabelEncoder: {e}")
        return "Error de Carga", (None, None)

    # Convertir PPM a float y calcular centroide
    try:
        h2_ppm_f = float(h2_ppm); ch4_ppm_f = float(ch4_ppm); c2h2_ppm_f = float(c2h2_ppm)
        c2h4_ppm_f = float(c2h4_ppm); c2h6_ppm_f = float(c2h6_ppm)
    except ValueError:
        return "Error: Valores de PPM deben ser numéricos", (None, None)

    Cx, Cy = calculate_centroid_from_ppm(h2_ppm_f, ch4_ppm_f, c2h2_ppm_f, c2h4_ppm_f, c2h6_ppm_f)

    if not (isinstance(Cx, (int, float)) and isinstance(Cy, (int, float))):
         return "Error en cálculo de centroide (no numérico)", (Cx, Cy)

    # Preparar datos para la predicción
    centroid_data_df = pd.DataFrame([[Cx, Cy]], columns=features)
    
    # Realizar la predicción
    try:
        with warnings.catch_warnings(): # Suprimir el UserWarning sobre feature names si aún persiste
            warnings.simplefilter("ignore", UserWarning)
            predicted_label_numeric = ml_model.predict(centroid_data_df)
        
        predicted_fault_zone_text = le.inverse_transform(predicted_label_numeric)
        return predicted_fault_zone_text[0], (Cx, Cy)
    except Exception as e:
        print(f"Error durante la predicción del modelo: {e}")
        return "Error en predicción", (Cx, Cy)

# --- 3. Ejecución Principal ---
if __name__ == "__main__":
    # Datos de ejemplo (puedes cambiarlos o pedirlos al usuario)
    h2_input_ppm = 15806
    ch4_input_ppm = 15355
    c2h2_input_ppm = 0
    c2h4_input_ppm = 12
    c2h6_input_ppm = 9290

    print(f"--- Diagnóstico para la Muestra de Gases ---")
    print(f"Valores de entrada (ppm): H2={h2_input_ppm}, CH4={ch4_input_ppm}, C2H2={c2h2_input_ppm}, C2H4={c2h4_input_ppm}, C2H6={c2h6_input_ppm}")

    predicted_zone, (cx_calc, cy_calc) = diagnose_dga_sample(
        h2_input_ppm, ch4_input_ppm, c2h2_input_ppm,
        c2h4_input_ppm, c2h6_input_ppm,
        MODEL_FILENAME, LABEL_ENCODER_FILENAME, FEATURE_NAMES
    )

    if cx_calc is not None and cy_calc is not None:
        print(f"\nCoordenadas del Centroide Calculadas: Cx = {cx_calc:.2f}, Cy = {cy_calc:.2f}")
    else:
        print("\nNo se pudieron calcular las coordenadas del centroide.")
    
    print(f"Zona de Falla Predicha: {predicted_zone}")

  
# --- Sección de Graficación ---
    plot_condition_met = False
    if isinstance(cx_calc, (int, float)) and isinstance(cy_calc, (int, float)):
        if isinstance(predicted_zone, str) and "Error" not in predicted_zone.lower(): # Verificar que no sea un mensaje de error
            plot_condition_met = True
        elif not isinstance(predicted_zone, str): # Si es una etiqueta de clase normal (no un string de error)
             plot_condition_met = True

    if plot_condition_met:
        print("\nGenerando gráfica del centroide...")
        try:
            plt.figure(figsize=(8, 8))
            # Vértices del Pentágono Exterior (Escala 40% para graficar)
            P_H2_plot = (0, 40); P_C2H6_plot = (-38, 12.4); P_CH4_plot = (-23.5, -32.4)
            P_C2H4_plot = (23.5, -32.4); P_C2H2_plot = (38, 12.4)
            pentagon_outer_plot_coords = [P_H2_plot, P_C2H6_plot, P_CH4_plot, P_C2H4_plot, P_C2H2_plot, P_H2_plot]
            x_outer_plot, y_outer_plot = zip(*pentagon_outer_plot_coords)
            plt.plot(x_outer_plot, y_outer_plot, 'k-', linewidth=0.7, alpha=0.6, label='_nolegend_') # Contorno
            
            # Aseguramos que cx_calc y cy_calc son flotantes para las operaciones de ploteo
            if cx_calc is None or cy_calc is None:
                raise ValueError("Las coordenadas del centroide no pueden ser None para graficar.")
            cx_plot = float(cx_calc)
            cy_plot = float(cy_calc)

            plt.scatter([cx_plot], [cy_plot], color='red', s=120, zorder=5, edgecolors='black', label=f'Centroide: {predicted_zone}')
            plt.text(cx_plot + 1, cy_plot + 1, f"{predicted_zone}\n({cx_plot:.1f}, {cy_plot:.1f})", fontsize=10, va='bottom', ha='left')

            plt.xlabel("Coordenada X", fontsize=12); plt.ylabel("Coordenada Y", fontsize=12)
            plt.title("Ubicación del Centroide Calculado en el Pentágono de Duval", fontsize=14)
            plt.axhline(0, color='grey', lw=0.5, linestyle=':'); plt.axvline(0, color='grey', lw=0.5, linestyle=':')
            plt.grid(True, linestyle=':', alpha=0.5); plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-45, 45); plt.ylim(-40, 45); plt.legend(loc='upper right'); plt.show()
        except Exception as e:
            print(f"Error al generar la gráfica: {e}")
    elif "Error" in str(predicted_zone): # Si predicted_zone es un mensaje de error
        print(f"No se grafica el punto debido a error en predicción/cálculo: {predicted_zone}")
    else: # Si cx_calc o cy_calc son None
        print(f"No se grafica el punto, coordenadas del centroide no válidas: Cx={cx_calc}, Cy={cy_calc}")