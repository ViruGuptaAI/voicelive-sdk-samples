# Create virtual environment if it doesn't exist and install requirements
if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\activate"
pip install -r scripts/requirements_setup_intvect.txt --quiet
python scripts\setup_intvect.py