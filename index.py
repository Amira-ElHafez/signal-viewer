from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Import the app instance
from app import app 

# --- IMPORT ALL PAGE LAYOUTS ---
from pages.Drone import drone 
from pages.EEG import eeg
from pages.ECG import ecg 

# --- Header Layout ---
header = dbc.Navbar(
    [
    dbc.Container([
        dbc.NavbarBrand([
            html.I(className="fas fa-chart-pie me-2", style={'font-size': '1.5rem'}),
            "Signal Viewer Pro"
        ], className="fw-bold", style={'font-size': '1.4rem', 'color': 'white'}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-home me-1"), " Home"], href="/", active="exact", className="nav-link-custom")),
            dbc.NavItem(dbc.NavLink([html.I(className="fa-solid fa-satellite-dish me-1"), " Drone Detection"], href="/drone", active="exact", className="nav-link-custom")),
            dbc.NavItem(dbc.NavLink([html.I(className="fa-solid fa-brain me-1"), " EEG Viewer"], href="/eeg", active="exact", className="nav-link-custom")),
            
            # --- ADD THE NEW LINK TO THE ECG PAGE ---
            dbc.NavItem(dbc.NavLink([html.I(className="fa-solid fa-heart-pulse me-1"), " ECG Viewer"], href="/ecg", active="exact", className="nav-link-custom")), # <-- ADD THIS LINE
            
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-file-alt me-1"), " Reports"], href="/reports", active="exact", className="nav-link-custom")),
            dbc.NavItem(dbc.NavLink([html.I(className="fas fa-cog me-1"), " Settings"], href="/settings", active="exact", className="nav-link-custom")),
        ], className="ms-auto", navbar=True)
    ], fluid=True)
    ], style={
        'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'padding': '1rem 0'
    }, className="mb-4")

# --- Main App Layout ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    header,
    dbc.Container(id="page-content", fluid=True, style={'padding': '0 2rem'})
])

# --- Page Navigation Callback ---
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/drone':
        return drone.layout
    elif pathname == '/eeg':
        return eeg.layout
    # --- ADD THE ROUTING RULE FOR THE ECG PAGE ---
    elif pathname == '/ecg': # <-- ADD THIS LINE
        return ecg.layout   # <-- ADD THIS LINE
    else: 
        return dbc.Container([
            html.H1("Welcome to the Signal Viewer Dashboard", className="mt-5"),
            html.P("Select a tool from the navigation bar above to begin analyzing signals.")
        ], className="text-center")

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
