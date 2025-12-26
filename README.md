# Mini-Pytorch3D

## Environment Setup

```bash
# Create project directory
git clone https://github.com/SimonXie2004/Mini-Pytorch3D

# If uv is not installed, download and install it
if ! command -v uv >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
uv venv --python 3.13 env_mini_pytorch3d
source env_mini_pytorch3d/bin/activate

# Deactivate conda environment if any
if command -v conda >/dev/null 2>&1; then
    conda deactivate 2>/dev/null || true
fi

# Install dependencies
uv pip install -r requirements.txt

## Download resources (data & external libraries)
bash scripts/download_resources.sh
```

## Build and Run

```bash
bash scripts/build_and_run.sh
```