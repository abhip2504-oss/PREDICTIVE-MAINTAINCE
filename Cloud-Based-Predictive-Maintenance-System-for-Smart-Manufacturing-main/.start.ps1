$ErrorActionPreference = "Stop"

# Use newly installed Python
$PythonPath = "C:\Users\Admin\AppData\Local\Programs\Python\Python312"
$env:Path = "$PythonPath;$($PythonPath)\Scripts;$env:Path"

Write-Host "Creating Virtual Environment..."
python -m venv venv
.\venv\Scripts\activate

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Training Machine Learning Model..."
python train_model.py

Write-Host "Starting simulated IoT Sensor Stream in background..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "sensor_stream.py"

Write-Host "Starting Streamlit Dashboard..."
# Streamlit will automatically open the browser on windows
streamlit run dashboard.py
