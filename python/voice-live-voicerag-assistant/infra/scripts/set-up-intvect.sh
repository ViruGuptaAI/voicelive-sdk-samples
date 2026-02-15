# Create virtual environment if it doesn't exist and install requirements
if [ ! -d "./.venv" ]; then
    python -m venv .venv
fi
source ./.venv/bin/activate
pip install -r scripts/requirements_setup_intvect.txt --quiet
python scripts/setup_intvect.py