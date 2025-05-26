import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import os
import warnings

# --- Nombres de Archivos del Modelo y LabelEncoder ---
MODEL_FILENAME = 'modelo_pentagono_rf.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder_pentagono.pkl'
FEATURE_NAMES = ['x_centroid', 'y_centroid']

# --- Cargar Modelo y LabelEncoder ---
MODEL_ABS_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
LABEL_ENCODER_ABS_PATH = os.path.join(os.getcwd(), LABEL_ENCODER_FILENAME)
ml_model, label_encoder, model_load_error = None, None, ""
try:
    if os.path.exists(MODEL_ABS_PATH) and os.path.exists(LABEL_ENCODER_ABS_PATH):
        ml_model = joblib.load(MODEL_ABS_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_ABS_PATH)
        print(f"Modelo ('{MODEL_FILENAME}') y LabelEncoder ('{LABEL_ENCODER_FILENAME}') cargados.")
    else:
        model_load_error = f"Error: No se encontraron '{MODEL_FILENAME}' o '{LABEL_ENCODER_FILENAME}' en {os.getcwd()}"
        print(model_load_error)
except Exception as e:
    model_load_error = f"Error al cargar modelo/encoder: {e}"
    print(model_load_error)

# --- Definiciones del Pentágono ---
SUMMIT_COORDS_100_SCALE = {
    "H2": (0, 100.0), "C2H6": (-95.10565, 30.9017), "CH4": (-58.7785, -80.9017),
    "C2H4": (58.7785, -80.9017), "C2H2": (95.10565, 30.9017)
}
GAS_ORDER_FOR_POLYGON = ["H2", "C2H6", "CH4", "C2H4", "C2H2"]
P_H2_PLOT = (0, 40); P_C2H6_PLOT = (-38, 12.4); P_CH4_PLOT = (-23.5, -32.4)
P_C2H4_PLOT = (23.5, -32.4); P_C2H2_PLOT = (38, 12.4)
PENTAGON_OUTER_PLOT_COORDS = [P_H2_PLOT, P_C2H6_PLOT, P_CH4_PLOT, P_C2H4_PLOT, P_C2H2_PLOT, P_H2_PLOT]
ALL_ZONES_VERTICES = { # Asegúrate que estas sean tus coordenadas finales y correctas
    "PD": [(-1, 33), (0, 33), (0, 24.5), (-1, 24.5), (-1, 33)],
    "S":  [(-38, 12.4), (-35, 3.1), (0, 1.5), (0, 24.5), (-1, 24.5), (-1,33), (0,33), (0,40), (-38, 12.4)],
    "T1-O": [(-35, 3.1), (-23.5, -32.4), (-22.5, -32.4), (-18.64, -25.76), (-11, -8), (-6, -4), (0, 1.5), (-35, 3.1)],
    "T2-O": [(-22.5, -32.4), (-23.5, -32.4), (-11,-8), (-22.5, -32.4)],
    "T2-C": [(-21.5, -32.4), (-18.64, -25.76), (-11,-8), (-6, -4), (-3.5,-3.5), (-21.5, -32.4)],
    "T3-C": [(-21.5, -32.4), (-3.5, -3.5), (0,-3), (1, -32.4), (-21.5, -32.4)],
    "T1-C": [(-18.64, -25.76), (-11, -8), (-6, -4), (-3.5,-3.5), (-18.64, -25.76)],
    "T3-H": [(1, -32.4), (23.5, -32.4), (24.3, -30), (0, -3), (1, -32.4)],
    "D2": [(0, 1.5), (-6,-4), (-3.5,-3.5), (0,-3), (24.3, -30), (32, -6.1), (4,16), (0,1.5)],
    "D1": [(0, 1.5), (4,16), (32, -6.1), (38, 12.4), (0,40), (0,1.5)]
}
ZONE_COLORS = { # RGBA para control de opacidad del relleno
    "PD": "rgba(0, 255, 255, 0.5)",   # Cyan más intenso
    "S":  "rgba(127, 255, 0, 0.5)",   # Chartreuse / Verde lima
    "T1-O": "rgba(255, 127, 80, 0.5)", # Coral
    "T2-O": "rgba(30, 144, 255, 0.5)",  # DodgerBlue
    "T2-C": "rgba(255, 140, 0, 0.5)",   # DarkOrange
    "T3-C": "rgba(255, 20, 147, 0.5)",  # DeepPink
    "T1-C": "rgba(186, 85, 211, 0.5)",  # MediumOrchid
    "T3-H": "rgba(160, 82, 45, 0.5)",   # Sienna (Marrón más rojizo)
    "D2": "rgba(112, 128, 144, 0.5)", # SlateGray
    "D1": "rgba(255, 215, 0, 0.5)"   # Gold
}
# Colores de texto para las etiquetas de zona, buscando contraste
ZONE_TEXT_COLORS = {
    "PD": "black", "S": "black", "T1-O": "black", "T2-O": "black",
    "T2-C": "black", "T3-C": "black", "T1-C": "black", "T3-H": "white", # Blanco para T3-H (marrón oscuro)
    "D2": "white", "D1": "black" # Blanco para D2 (gris oscuro)
}


# --- Funciones de Cálculo y Predicción (sin cambios, asumimos que están correctas) ---
def calculate_duval_centroid_from_percentages(gas_percentages_dict):
    polygon_vertices = []
    for gas_name in GAS_ORDER_FOR_POLYGON:
        percentage = gas_percentages_dict.get(gas_name, 0.0)
        summit_x, summit_y = SUMMIT_COORDS_100_SCALE[gas_name]
        polygon_vertices.append((percentage * summit_x, percentage * summit_y))
    pv = np.array(polygon_vertices); x, y = pv[:, 0], pv[:, 1]
    A_duval_term = sum(x[i] * y[(i + 1) % 5] - x[(i + 1) % 5] * y[i] for i in range(5))
    A_duval_calc = 0.5 * A_duval_term
    if abs(A_duval_calc) < 1e-9: return (0.0,0.0) if all(abs(val)<1e-9 for val in gas_percentages_dict.values()) else (np.mean(x),np.mean(y))
    cx_sum_term = sum((x[i]+x[(i+1)%5])*(x[i]*y[(i+1)%5]-x[(i+1)%5]*y[i]) for i in range(5))
    cy_sum_term = sum((y[i]+y[(i+1)%5])*(x[i]*y[(i+1)%5]-x[(i+1)%5]*y[i]) for i in range(5))
    return (1/(6*A_duval_calc))*cx_sum_term, (1/(6*A_duval_calc))*cy_sum_term

def calculate_centroid_from_ppm(h2, ch4, c2h2, c2h4, c2h6):
    gases_ppm = {"H2":h2,"CH4":ch4,"C2H2":c2h2,"C2H4":c2h4,"C2H6":c2h6}
    total_ppm = sum(gases_ppm.values())
    if total_ppm == 0: return 0.0, 0.0
    gas_percentages = {name:ppm/total_ppm for name,ppm in gases_ppm.items()}
    return calculate_duval_centroid_from_percentages(gas_percentages)

def predict_fault_zone(h2, ch4, c2h2, c2h4, c2h6):
    if ml_model is None or label_encoder is None: return "Error: Modelo no cargado",(None,None),model_load_error
    try: h2_f,ch4_f,c2h2_f,c2h4_f,c2h6_f = float(h2),float(ch4),float(c2h2),float(c2h4),float(c2h6)
    except (ValueError,TypeError): return "Error: PPMs deben ser numéricos.",(None,None),""
    Cx,Cy = calculate_centroid_from_ppm(h2_f,ch4_f,c2h2_f,c2h4_f,c2h6_f)
    if not (isinstance(Cx,(int,float)) and isinstance(Cy,(int,float))): return "Error cálculo centroide.",(Cx,Cy),""
    centroid_data_df = pd.DataFrame([[Cx,Cy]], columns=FEATURE_NAMES)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            predicted_label_numeric = ml_model.predict(centroid_data_df)
        predicted_fault_zone_text = label_encoder.inverse_transform(predicted_label_numeric)
        return predicted_fault_zone_text[0],(Cx,Cy),""
    except Exception as e: return f"Error predicción: {e}",(Cx,Cy),""

# --- Funciones para Crear Figuras Plotly ---
def create_pentagon_figure_with_zones(annotations_for_centroid=None): # Renombrado para claridad
    fig = go.Figure()
    x_outer, y_outer = zip(*PENTAGON_OUTER_PLOT_COORDS)
    fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='black', width=2.5), name='Pentágono Exterior', hoverinfo='skip')) # Borde más grueso
    
    zone_annotations = [] 
    for zone_name, vertices in ALL_ZONES_VERTICES.items():
        if not vertices or len(vertices) < 3: continue
        closed_vertices = list(vertices); 
        if closed_vertices[0] != closed_vertices[-1]: closed_vertices.append(closed_vertices[0])
        vx, vy = zip(*closed_vertices)
        fig.add_trace(go.Scatter(x=vx,y=vy,mode='lines',line=dict(width=1,color='rgba(0,0,0,0.7)'),fill='toself', # Borde de zona más grueso
                                 fillcolor=ZONE_COLORS.get(zone_name,'rgba(128,128,128,0.3)'),name=zone_name,hoverinfo='name'))
        if vertices:
            center_x, center_y = np.mean([v[0] for v in vertices]), np.mean([v[1] for v in vertices])
            # Ajustes de posición para etiquetas de zona (pueden necesitar más ajustes finos)
            if zone_name=="PD": center_y-=2.5
            if zone_name=="S": center_x+=6; center_y+=2.5
            if zone_name=="T1-C": center_y+=2; center_x-=2
            if zone_name=="T2-O": center_y+=2; center_x+=2
            if zone_name=="D1": center_x-=4; center_y-=3
            if zone_name=="D2": center_x+=4; center_y+=3
            if zone_name=="T2-C": center_y-=1 # Ajuste para T2-C
            if zone_name=="T3-C": center_y+=1 # Ajuste para T3-C
            
            zone_annotations.append(dict(x=center_x,y=center_y,text=f"<b>{zone_name}</b>",showarrow=False,
                                       font=dict(size=13, color=ZONE_TEXT_COLORS.get(zone_name, "black"), family="Arial Black, sans-serif"), # Fuente más grande, negrita y legible
                                       opacity=1.0)) # Opacidad completa para etiquetas de zona
                                       
    all_annotations = zone_annotations
    if annotations_for_centroid: # Si se pasan anotaciones para el centroide
        all_annotations.extend(annotations_for_centroid)

    fig.update_layout(
        title_text="Pentágono de Duval Combinado",
        xaxis=dict(visible=False, range=[-45, 45]), 
        yaxis=dict(visible=False, range=[-40, 45], scaleanchor="x", scaleratio=1), 
        showlegend=True,
        legend=dict(
            title_text='Zonas de Falla',
            font=dict(size=12, family="Arial, sans-serif"),
            bgcolor="rgba(240,240,240,0.8)", 
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1
        ),
        margin=dict(l=5, r=5, t=60, b=5), # Aumentar margen superior para el título
        plot_bgcolor='rgba(255,255,255,0)', 
        paper_bgcolor='rgba(255,255,255,0)', 
        annotations=all_annotations # Aplicar todas las anotaciones
    )
    return fig

def create_empty_pentagon_figure():
    # (Sin cambios en esta función, ya que solo muestra el contorno)
    fig = go.Figure()
    x_outer, y_outer = zip(*PENTAGON_OUTER_PLOT_COORDS)
    fig.add_trace(go.Scatter(x=x_outer,y=y_outer,mode='lines',line=dict(color='black',width=2.5),name='Pentágono Exterior',hoverinfo='skip'))
    fig.update_layout(
        title_text="Pentágono de Duval - Ingrese Datos",
        xaxis=dict(visible=False, range=[-45, 45]), 
        yaxis=dict(visible=False, range=[-40, 45], scaleanchor="x", scaleratio=1), 
        showlegend=False, margin=dict(l=5, r=5, t=60, b=5),
        plot_bgcolor='rgba(255,255,255,0)', paper_bgcolor='rgba(255,255,250,0)'
    )
    return fig

# --- Aplicación Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR]) 
app.layout = dbc.Container([ # ... (Layout sin cambios significativos, solo los inputs) ...
    dbc.Row(dbc.Col(html.H1("Diagnóstico de Fallas en Transformadores con Pentágono de Duval y ML", className="text-center text-primary mb-4"),width=12)),
    dbc.Row([
        dbc.Col([
            html.H5("Ingresar Concentraciones de Gases (ppm):",className="mb-3"),
            dbc.InputGroup([dbc.InputGroupText("H2"), dbc.Input(id="h2-ppm",type="number", placeholder="Ej: 50")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("CH4"), dbc.Input(id="ch4-ppm",type="number", placeholder="Ej: 20")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H2"), dbc.Input(id="c2h2-ppm",type="number", placeholder="Ej: 5")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H4"), dbc.Input(id="c2h4-ppm",type="number", placeholder="Ej: 30")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H6"), dbc.Input(id="c2h6-ppm",type="number", placeholder="Ej: 15")]),html.Br(),
            dbc.Button("Diagnosticar Falla",id="diagnose-button",color="primary",className="mt-3",n_clicks=0,style={'width':'100%'}),
            html.Div(id="error-message-output",className="mt-3 text-danger")
        ], md=4, className="bg-light p-4 rounded shadow-sm"),
        dbc.Col([
            dcc.Graph(id="pentagon-graph", style={'height':'65vh'}), # Un poco más de altura
            html.Div([html.H5("Resultado:",className="mt-4"), dbc.Alert(id="diagnosis-result-output",color="info",className="mt-2",style={'textAlign':'center','fontSize':'1.2em'})])
        ], md=8)
    ], className="mt-4"),
    dbc.Row(dbc.Col(html.Footer("App Duval ML",className="text-center text-muted mt-5 mb-3"),width=12))
], fluid=True, className="py-5")


@app.callback(
    [Output("pentagon-graph","figure"), Output("diagnosis-result-output","children"),
     Output("diagnosis-result-output","color"), Output("error-message-output","children")],
    [Input("diagnose-button","n_clicks")],
    [State("h2-ppm","value"), State("ch4-ppm","value"), State("c2h2-ppm","value"),
     State("c2h4-ppm","value"), State("c2h6-ppm","value")]
)
def update_diagnosis(n_clicks, h2, ch4, c2h2, c2h4, c2h6):
    if n_clicks == 0:
        return create_empty_pentagon_figure(), "Ingrese datos y presione 'Diagnosticar'.", "light", model_load_error if model_load_error else ""
    
    if model_load_error: 
        return create_empty_pentagon_figure(), "Error Crítico al Cargar Modelo.", "danger", model_load_error

    inputs = [h2, ch4, c2h2, c2h4, c2h6]
    if any(val is None for val in inputs):
        current_fig = create_pentagon_figure_with_zones() 
        return current_fig, "Error: Todos los campos de gases deben tener un valor numérico.", "warning", ""
    
    predicted_zone, (cx,cy), error_detail = predict_fault_zone(*inputs)
    
    fig_annotations_for_centroid = [] # Lista para la anotación del centroide (si la queremos)
    alert_color, diagnosis_text, error_msg_display = "info", "", error_detail or model_load_error

    # Crear la figura base con las zonas y sus etiquetas
    fig = create_pentagon_figure_with_zones() 

    if "Error" in str(predicted_zone):
        diagnosis_text, alert_color = predicted_zone, "danger"
        # No se añade marcador de centroide si hay error
    elif cx is not None and cy is not None:
        # Añadir SOLO el marcador del centroide (sin la anotación de texto con fondo)
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode='markers',
            marker=dict(
                color='black', # Color del marcador del centroide
                size=20,      # Tamaño del marcador aumentado
                symbol='diamond-tall', # Símbolo de diamante
                line=dict(width=2, color='white') # Borde blanco para destacar
            ),
            name='Centroide Calculado', # Aparecerá en la leyenda si showlegend=True para este trace
            hoverinfo='text',
            text=f'<b>{predicted_zone}</b><br>X: {cx:.1f}, Y: {cy:.1f}', # Info al pasar el mouse
            showlegend=False # No mostrar este punto específico en la leyenda principal de zonas
        ))
        diagnosis_text = f"Zona Predicha: {predicted_zone} (Cx={cx:.2f}, Cy={cy:.2f})"
        alert_color = "success"
    else:
        diagnosis_text, alert_color = "No se pudo calcular centroide/predicción.", "warning"

    if model_load_error and not error_msg_display: error_msg_display = model_load_error
    if diagnosis_text == "Error: Modelo no cargado": diagnosis_text = "" 
        
    return fig, diagnosis_text, alert_color, error_msg_display

if __name__ == '__main__':
    app.run(debug=True)
