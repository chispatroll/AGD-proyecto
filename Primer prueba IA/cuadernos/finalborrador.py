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
    "PD": [
        (-1, 33), (0, 33), (0, 24.5), (-1, 24.5),
        (-1, 33)
    ],
    "S": [
        (-38, 12.4),
        (-35, 3.1),
        (0, 1.5),
        (0, 24.5),
        (-1, 24.5),
        (-1, 33),
        (0, 33),
        (0, 40),
        (-38, 12.4)
    ],
    "T1-O": [
        (-35, 3.1),
        (-23.5, -32.4),
        (-22.5, -32.4),
        (-18.64, -25.76),
        (-11, -8),
        (-6, -4),
        (0, -3),
        (0, 1.5),
        (-35, 3.1)
    ],
    "T2-O": [
        (-22.5, -32.4),
        (-21.5, -32.4),
        (-18.64, -25.76),
        (-22.5, -32.4)
    ],
    "T2-C": [
        (-21.5, -32.4),
        (-18.64, -25.76),
        (-6, -4),
        (1, -32.4),
        (-21.5, -32.4)
    ],
    "T3-C": [
        (-6,-4),
        (-3.5, -3.5),
        (2.5, -32.4),
        (1, -32.4),
        (-6,-4)
    ],
    "T1-C": [
       (-18.64, -25.76),
        (-11, -8),
        (-6, -4),
       (-18.64, -25.76)
    ],
    "T3-H": [
        (-3.5, -3.5),
        (0, -3),
        (24.3, -30),
        (23.5, -32.4),
        (2.3, -32.4),
        (-3.5, -3.5)
    ],
    "D2": [
       (0, -3),
       (0, 1.5),
       (4, 16),
       (32, -6.1),
       (24.3, -30),
       (0, -3)
    ],
    "D1": [
       (0, 40),
       (38, 12.4),
       (32, -6.1),
       (4, 16),
       (0, 1.5),
       (0, 40)
    ]
}
ZONE_COLORS = {
    "PD": "rgba(0,255,255,0.4)","S":"rgba(144,238,144,0.4)","T1-O":"rgba(255,160,122,0.4)",
    "T2-O":"rgba(135,206,250,0.4)","T2-C":"rgba(255,165,0,0.4)","T3-C":"rgba(255,105,180,0.4)",
    "T1-C":"rgba(221,160,221,0.4)","T3-H":"rgba(139,69,19,0.4)","D2":"rgba(169,169,169,0.4)",
    "D1":"rgba(240,230,140,0.4)"
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
def create_pentagon_figure_with_zones(annotations_to_add=None):
    fig = go.Figure()
    x_outer, y_outer = zip(*PENTAGON_OUTER_PLOT_COORDS)
    fig.add_trace(go.Scatter(x=x_outer, y=y_outer, mode='lines', line=dict(color='black', width=2), name='Pentágono Exterior', hoverinfo='skip'))
    
    current_annotations = [] 
    for zone_name, vertices in ALL_ZONES_VERTICES.items():
        if not vertices or len(vertices) < 3: continue
        closed_vertices = list(vertices); 
        if closed_vertices[0] != closed_vertices[-1]: closed_vertices.append(closed_vertices[0])
        vx, vy = zip(*closed_vertices)
        fig.add_trace(go.Scatter(x=vx,y=vy,mode='lines',line=dict(width=0.5,color='rgba(0,0,0,0.6)'),fill='toself',
                                 fillcolor=ZONE_COLORS.get(zone_name,'rgba(128,128,128,0.3)'),name=zone_name,hoverinfo='name'))
        if vertices:
            center_x, center_y = np.mean([v[0] for v in vertices]), np.mean([v[1] for v in vertices])
            if zone_name=="PD": center_y-=2
            if zone_name=="S": center_x+=5; center_y+=2
            if zone_name=="T1-C": center_y+=1; center_x-=1
            if zone_name=="T2-O": center_y+=1; center_x+=1
            if zone_name=="D1": center_x-=2; center_y-=2
            if zone_name=="D2": center_x+=2; center_y+=2
            current_annotations.append(dict(x=center_x,y=center_y,text=f"<b>{zone_name}</b>",showarrow=False,
                                       font=dict(size=9,color="black" if zone_name not in ["T3-H","D2"] else "white"),opacity=0.85))
    if annotations_to_add: current_annotations.extend(annotations_to_add)

    fig.update_layout(
        title_text="Pentágono de Duval Combinado",
        # Ocultar ejes X e Y completamente
        xaxis=dict(visible=False, range=[-45, 45]), # Mantener el rango para escala interna
        yaxis=dict(visible=False, range=[-40, 45], scaleanchor="x", scaleratio=1), # Mantener escala y aspecto
        showlegend=True,legend_title_text='Zonas de Falla',
        margin=dict(l=5, r=5, t=50, b=5), # Reducir márgenes si los ejes están ocultos
        plot_bgcolor='rgba(255,255,255,0)', # Fondo del gráfico transparente
        paper_bgcolor='rgba(255,255,255,0)', # Fondo del papel transparente (para exportación)
        annotations=current_annotations
    )
    return fig

def create_empty_pentagon_figure():
    fig = go.Figure()
    x_outer, y_outer = zip(*PENTAGON_OUTER_PLOT_COORDS)
    fig.add_trace(go.Scatter(x=x_outer,y=y_outer,mode='lines',line=dict(color='black',width=2),name='Pentágono Exterior',hoverinfo='skip'))
    fig.update_layout(
        title_text="Pentágono de Duval - Ingrese Datos",
        xaxis=dict(visible=False, range=[-45, 45]), # Ocultar eje X
        yaxis=dict(visible=False, range=[-40, 45], scaleanchor="x", scaleratio=1), # Ocultar eje Y
        showlegend=False,
        margin=dict(l=5, r=5, t=50, b=5),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)'
    )
    return fig

# --- Aplicación Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR]) # O el tema que prefieras
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Diagnóstico de Fallas en Transformadores con Pentágono de Duval y ML", className="text-center text-primary mb-4"),width=12)),
    dbc.Row([
        dbc.Col([
            html.H5("Ingresar Concentraciones de Gases (ppm):",className="mb-3"),
            dbc.InputGroup([dbc.InputGroupText("H2"), dbc.Input(id="h2-ppm",type="number")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("CH4"), dbc.Input(id="ch4-ppm",type="number")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H2"), dbc.Input(id="c2h2-ppm",type="number")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H4"), dbc.Input(id="c2h4-ppm",type="number")]),html.Br(),
            dbc.InputGroup([dbc.InputGroupText("C2H6"), dbc.Input(id="c2h6-ppm",type="number")]),html.Br(),
            dbc.Button("Diagnosticar Falla",id="diagnose-button",color="primary",className="mt-3",n_clicks=0,style={'width':'100%'}),
            html.Div(id="error-message-output",className="mt-3 text-danger")
        ], md=4, className="bg-light p-4 rounded shadow-sm"),
        dbc.Col([
            dcc.Graph(id="pentagon-graph", style={'height':'60vh'}), # Ajusta la altura si es necesario
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
    # Usar dash.no_update para el estado inicial si no hay valores por defecto o n_clicks es 0
    # Opcionalmente, puedes llamar a create_empty_pentagon_figure() si prefieres mostrarlo vacío.
    if n_clicks == 0 and not any(s is not None for s in [h2, ch4, c2h2, c2h4, c2h6]):
        return create_empty_pentagon_figure(), "Ingrese datos y presione 'Diagnosticar'.", "light", model_load_error if model_load_error else ""
    
    if model_load_error:
        return create_empty_pentagon_figure(), "Error Crítico al Cargar Modelo.", "danger", model_load_error

    inputs = [h2,ch4,c2h2,c2h4,c2h6]
    if any(val is None for val in inputs): # Chequeo si algún campo está vacío
        # Devolver la figura actual sin cambios si hay un error de entrada
        current_fig = create_pentagon_figure_with_zones() # O la figura vacía
        return current_fig, "Error: Todos los campos deben tener valor.", "warning", ""
    
    predicted_zone, (cx,cy), error_detail = predict_fault_zone(*inputs)
    
    fig_annotations = [] 
    alert_color, diagnosis_text, error_msg_display = "info", "", error_detail or model_load_error

    if "Error" in str(predicted_zone):
        diagnosis_text, alert_color = predicted_zone, "danger"
    elif cx is not None and cy is not None:
        fig_annotations.append(dict(x=cx,y=cy,text=f"<b>{predicted_zone}</b><br>({cx:.1f},{cy:.1f})",showarrow=True,arrowhead=2,
                                  arrowcolor="black",arrowsize=1,arrowwidth=2,
                                  ax=cx+np.sign(cx if cx!=0 else 1)*5 if abs(cx)<30 else cx-np.sign(cx)*10,
                                  ay=cy+np.sign(cy if cy!=0 else 1)*5 if abs(cy)<30 else cy-np.sign(cy)*10,
                                  font=dict(size=10,color="black"),bgcolor="rgba(255,255,255,0.7)",bordercolor="black",borderwidth=1))
        diagnosis_text = f"Zona Predicha: {predicted_zone} (Cx={cx:.2f}, Cy={cy:.2f})"
        alert_color = "success"
    else:
        diagnosis_text, alert_color = "No se pudo calcular centroide/predicción.", "warning"

    fig = create_pentagon_figure_with_zones(annotations_to_add=fig_annotations)
    
    if model_load_error and not error_msg_display: error_msg_display = model_load_error
    if diagnosis_text == "Error: Modelo no cargado": diagnosis_text = ""
        
    return fig, diagnosis_text, alert_color, error_msg_display

if __name__ == '__main__':
    app.run(debug=True)
