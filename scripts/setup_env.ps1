# setup_env.ps1 - Environment setup script for Windows

Write-Host "Creating Virtual Environment (.venv)..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Activating environment and installing dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Registering Jupyter Kernel..." -ForegroundColor Cyan
python -m ipykernel install --user --name=demography --display-name "Python (Demography)"

Write-Host "Setup Complete. You can now use the 'demography' kernel in VS Code or Jupyter." -ForegroundColor Green
