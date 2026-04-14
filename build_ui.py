import os
import time
import random
import subprocess
import sys

def launch():
    from IPython.display import display, HTML
    
    # 1. Create a dynamic display target natively at the very top of the output box which we can seamlessly overwrite later
    ui_container = display(HTML("<div style='padding: 10px; font-size: 18px; font-family: Arial; color: #333;'>⚙️ <b>Initializing backend environment...</b> This takes about ~15 seconds on your very first run. Please wait.</div>"), display_id=True)
    
    # 2. Quiet dependencies checks isolating keyring blocks natively
    env = os.environ.copy()
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    
    try:
        import torch
        import streamlit
        import networkx
    except ImportError:
        with open("pip_install.log", "w") as pip_log:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
                env=env, stdout=pip_log, stderr=subprocess.STDOUT
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit", "networkx", "plotly", "pandas"],
                env=env, stdout=pip_log, stderr=subprocess.STDOUT
            )
        
    # 3. Boot streamling securely handling ports and preventing jupyter freezes securely
    port = random.randint(15000, 30000)
    
    log_file = open('streamlit.log', 'w')
    process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py', 
         '--server.port', str(port), 
         '--server.address', '0.0.0.0',
         '--server.enableCORS', 'false',
         '--server.enableXsrfProtection', 'false',
         '--server.headless', 'true'],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )
    
    time.sleep(4)
    
    url = f"https://datahub.berkeley.edu/user-redirect/proxy/{port}/"
    button_html = f"""
    <div style="margin-top: 15px;">
        <a href="{url}" target="_blank" style="
            display: inline-block; 
            padding: 15px 32px; 
            font-size: 22px; 
            font-family: Arial, sans-serif;
            font-weight: bold; 
            color: white; 
            background-color: #003262;
            border-radius: 8px; 
            text-decoration: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">🚀 Launch MLP Visualizer</a>
        <p style="margin-top: 10px; font-size: 14px; color: #666; font-family: Arial;">(Make sure to disable pop-up blockers if nothing opens)</p>
    </div>
    """
    
    # Surgically overwrite the 'Initializing...' text block precisely with the newly injected button at the very top of the output
    ui_container.update(HTML(button_html))
