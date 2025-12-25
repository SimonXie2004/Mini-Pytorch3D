# Mini-Pytorch3D

## Environment Setup

```bash
# Create project directory
git clone https://github.com/SimonXie2004/Mini-Pytorch3D

# If you don't have uv, download and install first
wget -qO- https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.13 env_mini_pytorch3d
source env_mini_pytorch3d/bin/activate

# Deactivate conda environment if any
if command -v conda >/dev/null 2>&1; then
    conda deactivate 2>/dev/null || true
fi

# Install dependencies
uv pip install -r requirements.txt
```