# --- 1. Imports ---
import dash
from dash import dcc, html, Input, Output, State, no_update, exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import os
import base64
import io

# CHANGE 1: Import the main 'app' instance from app.py instead of creating a new one.
# This connects the "room" to the main "house".
from app import app

# --- 2. Global Settings & Cache ---
VIEW_SECONDS = 2.0
AUDIO_CACHE = {'y': None, 'sr': None}

# --- 3. Load the Drone Detection Model ---
try:
    print("Loading drone detection model from local folder...")
    
    # CHANGE 2: Correct the model path to be relative to the main app.py file.
    local_model_path = "./pages/Drone/drone_model"
    
    if not os.path.isdir(local_model_path):
        raise FileNotFoundError(
            f"Model folder not found at '{local_model_path}'. "
            "Please ensure this path is correct relative to where you run app.py."
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
    model = AutoModelForAudioClassification.from_pretrained(local_model_path)
    
    print("Drone detection model loaded successfully from local folder.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load the local model. {e}")
    model = None
    feature_extractor = None

# --- 4. Plotting Functions ---
def plot_spectrogram_plotly(y, sr):
    stft = librosa.stft(y)
    db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(db_spectrogram.shape[1]), sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    fig = go.Figure(data=go.Heatmap(
        z=db_spectrogram, x=times, y=freqs, colorscale='Viridis', colorbar={'title': 'Intensity (dB)'}
    ))
    fig.update_yaxes(type='log')
    # Use the 'plotly_white' template to match the main app's light theme.
    fig.update_layout(
        title="Spectrogram (Frequency Fingerprint)",
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", template="plotly_white"
    )
    return fig

# --- 5. Page Layout ---
# CHANGE 3: The layout is now a simple variable named 'layout', not 'app.layout'.
# Your main app.py will find this variable and display it.
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("🚁 Drone Sound Detector", className="text-center text-primary my-4"))),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-audio',
                children=html.Div(['Drag and Drop or ', html.A('Select an Audio File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                }
            ),
            html.Div(id='output-audio-player'),
            html.Br(),
            html.Div(id='output-prediction')
        ], width=12)
    ]),
    
    html.Div([
        html.Button('Start / Resume', id='start-button', disabled=True, className="btn btn-success me-2"),
        html.Button('Pause', id='pause-button', disabled=True, className="btn btn-warning me-2"),
        html.Button('Restart', id='restart-button', disabled=True, className="btn btn-danger me-2"),
    ], className="text-center my-3"),
    html.Hr(),
    dbc.Row([dbc.Col(dcc.Graph(id='waveform-graph'), width=12)]),
    dbc.Row([dbc.Col(dcc.Graph(id='spectrogram-graph'), width=12)]),

    dcc.Interval(id='graph-interval', interval=100, n_intervals=0, disabled=True),
    dcc.Store(id='position-store', data=0)

], fluid=True)


# --- 6. Callbacks ---
# These callbacks are now correctly registered to the main 'app' instance we imported.

@app.callback(
    Output('output-prediction', 'children'),
    Output('output-audio-player', 'children'),
    Output('waveform-graph', 'figure'),
    Output('spectrogram-graph', 'figure'),
    Output('start-button', 'disabled'),
    Output('pause-button', 'disabled'),
    Output('restart-button', 'disabled'),
    Output('graph-interval', 'disabled'),
    Output('position-store', 'data'),
    Input('upload-audio', 'contents')
)
def load_audio_and_classify(contents):
    print(f"--- RAW UPLOAD DATA (first 150 chars): {str(contents)[:150]}")
    
    if contents is None:
        raise exceptions.PreventUpdate

    if model is None or feature_extractor is None:
        error_alert = dbc.Alert("Error: Prediction model not loaded. Check server logs.", color="danger")
        return error_alert, None, go.Figure(), go.Figure(), True, True, True, True, 0

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        y, sr = librosa.load(io.BytesIO(decoded), sr=16000, mono=True)
        AUDIO_CACHE['y'] = y
        AUDIO_CACHE['sr'] = sr
    except Exception as e:
        print(f"AUDIO PROCESSING ERROR: {e}") 
        error_alert = dbc.Alert(f"Error processing audio file: {e}", color="danger")
        return error_alert, None, go.Figure(), go.Figure(), True, True, True, True, 0

    try:
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        
        scores = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        predicted_class_id = np.argmax(scores)
        predicted_label = model.config.id2label[predicted_class_id]
        confidence = scores[predicted_class_id] * 100
        
        alert_color = "success" if predicted_label == "drone" else "secondary"
        result_icon = "🚁" if predicted_label == "drone" else "✅"

        prediction_div = dbc.Alert(
            [
                html.H4(f"{result_icon} Prediction Result"),
                html.P(f"The sound is classified as: {predicted_label.upper()}", className="lead"),
                html.P(f"Confidence: {confidence:.2f}%")
            ],
            color=alert_color
        )

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        prediction_div = dbc.Alert(f"Error during prediction: {e}", color="danger")

    audio_player = html.Audio(src=contents, controls=True, style={'width': '100%'})
    spectrogram_fig = plot_spectrogram_plotly(y, sr)
    
    window_size_points = int(VIEW_SECONDS * sr)
    initial_chunk = y[:window_size_points]
    time_axis = np.linspace(0, VIEW_SECONDS, len(initial_chunk))
    waveform_fig = go.Figure(data=go.Scatter(x=time_axis, y=initial_chunk, mode='lines', line=dict(color='royalblue')))
    # Use 'plotly_white' template to match the light theme
    waveform_fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_white")

    return prediction_div, audio_player, waveform_fig, spectrogram_fig, False, False, False, True, 0


@app.callback(
    Output('graph-interval', 'disabled', allow_duplicate=True),
    Output('position-store', 'data', allow_duplicate=True),
    Output('waveform-graph', 'figure', allow_duplicate=True),
    Input('start-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    Input('restart-button', 'n_clicks'),
    prevent_initial_call=True
)
def manage_controls(start_clicks, pause_clicks, restart_clicks):
    triggered_id = dash.ctx.triggered_id
    y, sr = AUDIO_CACHE.get('y'), AUDIO_CACHE.get('sr')
    if y is None or sr is None:
        raise exceptions.PreventUpdate

    if triggered_id == 'start-button':
        return False, no_update, no_update
        
    if triggered_id == 'pause-button':
        return True, no_update, no_update
        
    if triggered_id == 'restart-button':
        window_size_points = int(VIEW_SECONDS * sr)
        initial_chunk = y[:window_size_points]
        time_axis = np.linspace(0, VIEW_SECONDS, len(initial_chunk))
        waveform_fig = go.Figure(data=go.Scatter(x=time_axis, y=initial_chunk, mode='lines', line=dict(color='royalblue')))
        waveform_fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_white")
        return True, 0, waveform_fig
        
    raise exceptions.PreventUpdate

@app.callback(
    Output('waveform-graph', 'figure', allow_duplicate=True),
    Output('position-store', 'data', allow_duplicate=True),
    Output('graph-interval', 'disabled', allow_duplicate=True),
    Input('graph-interval', 'n_intervals'),
    State('position-store', 'data'),
    prevent_initial_call=True
)
def update_waveform_scrolling(n, current_position):
    y, sr = AUDIO_CACHE.get('y'), AUDIO_CACHE.get('sr')
    if y is None or sr is None:
        raise exceptions.PreventUpdate
        
    window_size_points = int(VIEW_SECONDS * sr)
    step = int(window_size_points / 20)
    new_position = current_position + step

    if new_position + window_size_points > len(y):
        return no_update, no_update, True
        
    start_index = new_position
    end_index = start_index + window_size_points
    data_chunk = y[start_index:end_index]
    
    start_time_sec = new_position / sr
    end_time_sec = end_index / sr
    time_axis = np.linspace(start_time_sec, end_time_sec, len(data_chunk))
    
    fig = go.Figure(data=go.Scatter(x=time_axis, y=data_chunk, mode='lines', line=dict(color='royalblue')))
    fig.update_layout(
        title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
        xaxis=dict(range=[start_time_sec, end_time_sec]), template="plotly_white"
    )
    
    return fig, new_position, no_update



