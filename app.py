"""
Root Entrypoint for Streamlit Cloud and other deployment platforms.
Directs execution to stroke-ml-app/app.py with proper paths.
"""
import os
import sys
import runpy

# Ensure stroke-ml-app directory is in sys.path for internal imports (utils.*)
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "stroke-ml-app")

if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Execute main application script
target_script = os.path.join(app_dir, "app.py")
runpy.run_path(target_script, run_name="__main__")
