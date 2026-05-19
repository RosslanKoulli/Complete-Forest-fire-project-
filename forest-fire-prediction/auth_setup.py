"""
Authentication Setup
=====================
Adds login/password protection to the Streamlit app using
streamlit-authenticator.

Install: pip install streamlit-authenticator pyyaml

Setup:
1. Run generate_credentials() to create auth_config.yaml
2. Import setup_auth() in app.py and wrap your content

Usage in app.py:
    from auth_setup import setup_auth
    
    auth, name, status = setup_auth()
    if status:
        auth.logout('Logout', 'sidebar')
        # ... your app content ...
    elif status is False:
        st.error('Wrong username or password')
    else:
        st.info('Please log in')
"""

import os
import yaml


def generate_credentials(output_path: str = 'auth_config.yaml',
                          users: dict = None):
    """
    Generate an auth config file with hashed passwords.
    
    Parameters
    ----------
    output_path : str
        Where to save the YAML config
    users : dict, optional
        {username: {'name': str, 'email': str, 'password': str}}
        If None, creates default admin/demo accounts.
    
    Default accounts:
        admin / firepredict2026
        demo  / demo1234
    """
    try:
        import streamlit_authenticator as stauth
    except ImportError:
        print("Install streamlit-authenticator first:")
        print("  pip install streamlit-authenticator")
        return
    
    if users is None:
        users = {
            'admin': {
                'name': 'Admin User',
                'email': 'admin@brighton.ac.uk',
                'password': 'firepredict2026',
            },
            'demo': {
                'name': 'Demo User',
                'email': 'demo@brighton.ac.uk',
                'password': 'demo1234',
            },
        }
    
    # Hash all passwords
    passwords = [u['password'] for u in users.values()]
    hashed = stauth.Hasher(passwords).generate()
    
    credentials = {'usernames': {}}
    for i, (username, info) in enumerate(users.items()):
        credentials['usernames'][username] = {
            'email': info['email'],
            'name': info['name'],
            'password': hashed[i],
        }
    
    config = {
        'credentials': credentials,
        'cookie': {
            'expiry_days': 30,
            'key': 'forest_fire_prediction_cookie_key_2026',
            'name': 'fire_prediction_auth',
        },
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Auth config saved: {output_path}")
    print(f"Accounts created:")
    for username, info in users.items():
        print(f"  {username} / {info['password']}")


def setup_auth(config_path: str = 'auth_config.yaml'):
    """
    Initialise authentication in the Streamlit app.
    
    Returns
    -------
    authenticator : Authenticate object (for logout button)
    name : str or None (logged-in user's display name)
    authentication_status : bool or None
        True = logged in, False = wrong password, None = not yet tried
    """
    import streamlit as st
    
    if not os.path.exists(config_path):
        st.error(f"Auth config not found: {config_path}")
        st.info("Run `python auth_setup.py` to generate credentials.")
        return None, None, None
    
    try:
        import streamlit_authenticator as stauth
    except ImportError:
        st.error("streamlit-authenticator not installed.")
        st.code("pip install streamlit-authenticator pyyaml")
        return None, None, None
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    
    name, authentication_status, username = authenticator.login(
        'Login', 'main'
    )
    
    return authenticator, name, authentication_status


# ================================================================
# HOW TO INTEGRATE WITH app.py
# ================================================================
APP_INTEGRATION_EXAMPLE = """
# Add this at the TOP of app.py, after page config and styling:

from auth_setup import setup_auth

# Check if auth is enabled (config file exists)
import os
AUTH_ENABLED = os.path.exists('auth_config.yaml')

if AUTH_ENABLED:
    authenticator, name, auth_status = setup_auth()
    
    if auth_status:
        # Logged in — show logout button in sidebar
        authenticator.logout('Logout', 'sidebar')
        st.sidebar.write(f'Logged in as: {name}')
        
        # ===== YOUR MAIN APP CODE BELOW =====
        tab_predict, tab_compare, tab_simulate, tab_about = st.tabs([...])
        # ... rest of the app ...
        
    elif auth_status is False:
        st.error('Incorrect username or password.')
    else:
        st.info('Please log in to access the Forest Fire Prediction System.')
        st.markdown('**Demo account:** username `demo`, password `demo1234`')
else:
    # No auth config — run app without login
    tab_predict, tab_compare, tab_simulate, tab_about = st.tabs([...])
    # ... rest of the app ...
"""


if __name__ == '__main__':
    print("Generating authentication credentials...")
    print("=" * 50)
    
    try:
        generate_credentials()
        print("\nTo enable auth in the app, see auth_setup.py for integration example.")
    except Exception as e:
        print(f"\nCould not generate credentials: {e}")
        print("Install the package first:")
        print("  pip install streamlit-authenticator pyyaml")
        print("\nThe app works without auth — this is optional.")
