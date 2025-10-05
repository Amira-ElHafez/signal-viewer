# --- 1. Imports ---
import dash
from dash import dcc, html, Input, Output, State, no_update, exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch

import urllib.request
import os
import base64
import io

# --- 2. Global Settings & Cache ---
VIEW_SECONDS = 2.0
AUDIO_CACHE = {'y': None, 'sr': None}

# --- 3. Load the Drone Detection Model from Hugging Face ---
try:
    print("Loading drone detection model from Hugging Face...")
    # The model identifier from the Hugging Face Hub
    model_name = "preszzz/drone-audio-detection-05-17-trial-6"
    
    # The feature extractor prepares the audio data for the model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # The actual classification model
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    
    print("Drone detection model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load the Hugging Face model. {e}")
    model = None
    feature_extractor = None

# --- 4. Plotting Functions ---
def plot_spectrogram_plotly(y, sr):
    stft = librosa.stft(y)
    db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(db_spectrogram.shape[1]), sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    fig = go.Figure(data=go.Heatmap(
        z=db_spectrogram, x=times, y=freqs, colorscale='Inferno', colorbar={'title': 'Intensity (dB)'}
    ))
    fig.update_yaxes(type='log')
    fig.update_layout(
        title="Spectrogram (Frequency Fingerprint)",
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", template="plotly_dark"
    )
    return fig

# --- 5. Initialize the Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# --- 6. App Layout ---
# CHANGED: Updated the title to be more specific.
app.layout = dbc.Container([
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


# --- 7. Callbacks ---

# Callback 1: Handles file uploading, classification, and initial setup.
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
    if contents is None:
        raise exceptions.PreventUpdate

    if model is None or feature_extractor is None:
        error_alert = dbc.Alert("Error: Prediction model not loaded. Check server logs.", color="danger")
        return error_alert, None, go.Figure(), go.Figure(), True, True, True, True, 0

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # The new model was trained on 16000 Hz audio, so we resample.
        y, sr = librosa.load(io.BytesIO(decoded), sr=16000, mono=True)
        AUDIO_CACHE['y'] = y
        AUDIO_CACHE['sr'] = sr
    except Exception as e:
        error_alert = dbc.Alert(f"Error processing audio file: {e}", color="danger")
        return error_alert, None, go.Figure(), go.Figure(), True, True, True, True, 0

    # --- NEW: Drone Detection Prediction Logic ---
    try:
        # 1. Pre-process the audio using the feature extractor
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")

        # 2. Make a prediction with the model
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # 3. Convert logits to probabilities and get the highest score
        scores = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        predicted_class_id = np.argmax(scores)
        predicted_label = model.config.id2label[predicted_class_id]
        confidence = scores[predicted_class_id] * 100
        
        # Determine color and icon based on prediction
        alert_color = "success" if predicted_label == "drone" else "secondary"
        result_icon = "🚁" if predicted_label == "drone" else "✅"

        prediction_div = dbc.Alert(
            [
                html.H4(f"{result_icon} Prediction Result"),
                html.P(
                    f"The sound is classified as: {predicted_label.upper()}",
                    className="lead"
                ),
                html.P(f"Confidence: {confidence:.2f}%")
            ],
            color=alert_color
        )

    except Exception as e:
        prediction_div = dbc.Alert(f"Error during prediction: {e}", color="danger")


    # --- Create Initial Visuals (same as before) ---
    audio_player = html.Audio(src=contents, controls=True, style={'width': '100%'})
    spectrogram_fig = plot_spectrogram_plotly(y, sr)
    
    window_size_points = int(VIEW_SECONDS * sr)
    initial_chunk = y[:window_size_points]
    time_axis = np.linspace(0, VIEW_SECONDS, len(initial_chunk))
    waveform_fig = go.Figure(data=go.Scatter(x=time_axis, y=initial_chunk, mode='lines', line=dict(color='cyan')))
    waveform_fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark")

    return prediction_div, audio_player, waveform_fig, spectrogram_fig, False, False, False, True, 0

# Callback 2: Responds to the Start, Pause, and Restart buttons.
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
        waveform_fig = go.Figure(data=go.Scatter(x=time_axis, y=initial_chunk, mode='lines', line=dict(color='cyan')))
        waveform_fig.update_layout(title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", template="plotly_dark")
        return False, 0, waveform_fig
        
    raise exceptions.PreventUpdate

# Callback 3: Handles the scrolling animation triggered by the interval.
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
    
    fig = go.Figure(data=go.Scatter(x=time_axis, y=data_chunk, mode='lines', line=dict(color='cyan')))
    fig.update_layout(
        title="Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
        xaxis=dict(range=[start_time_sec, end_time_sec]), template="plotly_dark"
    )
    
    return fig, new_position, no_update

# --- 8. Run ---
if __name__ == '__main__':
    app.run(debug=True)

