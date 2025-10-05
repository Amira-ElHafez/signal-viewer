
# import dash
# from dash import dcc, html, Input, Output
# import dash_bootstrap_components as dbc
# from pages.Drone import drone 
# app = dash.Dash(__name__, 
#                 suppress_callback_exceptions=True, # Important for multi-page apps
#                 external_stylesheets=[
#                     dbc.themes.BOOTSTRAP, # Using a light theme to match your header style
#                     "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
#                 ])
# server = app.server

# # --- Header Layout ---
# # A beautiful gradient header with navigation links.
# header = dbc.Navbar([
#     dbc.Container([
#         # Brand/Logo section
#         dbc.NavbarBrand([
#             html.I(className="fas fa-chart-pie me-2", style={'font-size': '1.5rem'}),
#             "Signal Viewer Pro"
#         ], className="fw-bold", style={'font-size': '1.4rem', 'color': 'white'}),
        
#         # Navigation links
#         dbc.Nav([
#             dbc.NavItem(dbc.NavLink([
#                 html.I(className="fas fa-home me-1"), " Home"
#             ], href="/", active="exact", className="nav-link-custom")),
            
#             dbc.NavItem(dbc.NavLink([
#                 html.I(className="fa-solid fa-satellite-dish me-1"), " Drone Detection"
#             ], href="/drone", active="exact", className="nav-link-custom")),
            
#             dbc.NavItem(dbc.NavLink([
#                 html.I(className="fas fa-file-alt me-1"), " Reports"
#             ], href="/reports", active="exact", className="nav-link-custom")),
            
#             dbc.NavItem(dbc.NavLink([
#                 html.I(className="fas fa-cog me-1"), " Settings"
#             ], href="/settings", active="exact", className="nav-link-custom")),

#         ], className="ms-auto", navbar=True)
#     ], fluid=True)
# ], style={
#     'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
#     'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
#     'padding': '1rem 0'
# }, className="mb-4") # Added margin-bottom


# # --- Main App Layout ---
# # This layout contains the URL listener, the header, and the content area.
# app.layout = html.Div([
#     dcc.Location(id="url", refresh=False),
#     header,
#     dbc.Container(id="page-content", fluid=True, style={'padding': '0 2rem'}) # Main content area
# ])

# # --- Page Navigation Callback ---
# # This callback changes the content of 'page-content' based on the URL.
# @app.callback(
#     Output('page-content', 'children'),
#     Input('url', 'pathname')
# )
# def display_page(pathname):
#     if pathname == '/drone':
#         # When the user goes to /drone, we return the layout from our Drone.py file.
#         return drone.layout
    
#     # You can add other pages here later
#     # elif pathname == '/reports':
#     #     return reports.layout
    
#     # This is the default "Home" page content
#     else: 
#         return dbc.Container([
#             html.H1("Welcome to the Signal Viewer Dashboard", className="mt-5"),
#             html.P("Select a tool from the navigation bar above to begin analyzing signals.")
#         ], className="text-center")

# # --- Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True)
import dash
import dash_bootstrap_components as dbc

# This is the only thing this file does. It creates the app instance.
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ]
)

# Set the title that appears in the browser tab
app.title = "Signal Viewer Pro"

server = app.server