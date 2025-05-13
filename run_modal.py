import modal
import subprocess
import os

# Create a Modal Volume for caching the dataset
volume = modal.Volume.from_name("fineweb10b-cache", create_if_missing=True)

# Create a Modal Volume for storing training logs
logs_volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)

# Simple image for downloading data
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("data/requirements.txt")
    .workdir("/modded-nanogpt")
    .add_local_file(
        "data/cached_fineweb10B.py",
        remote_path="/modded-nanogpt/data/cached_fineweb10B.py",
    )
)

# Training image with CUDA support
train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install_from_requirements("requirements.txt")
    .workdir("/modded-nanogpt")
    .add_local_file("train_gpt.py", remote_path="/modded-nanogpt/train_gpt.py")
)


app = modal.App(name="modded-nanogpt")


@app.function(
    image=download_image, gpu=None, volumes={"/modded-nanogpt/data/fineweb10B": volume}
)
def prepare_dataset():
    # Check if dataset already exists in volume
    # We only need the first 8 chunks for this speedrun
    if not os.path.exists("/modded-nanogpt/data/fineweb10B/fineweb_train_000007.bin"):
        print("Dataset not found in volume, downloading...")
        cmd = "python data/cached_fineweb10B.py 8"
        subprocess.run(cmd, shell=True)
        print("Dataset downloaded and cached in volume")
    else:
        print("Dataset already exists in volume")


@app.function(
    image=train_image,
    gpu="H100:8",
    volumes={
        "/modded-nanogpt/data/fineweb10B": volume,
        "/modded-nanogpt/logs": logs_volume,
    },
    # timeout is 20 minutes, training should only take 3 minutes but warmup etc.
    timeout=20 * 60,
)
def fit():
    cmd = "torchrun --standalone --nproc_per_node=8 train_gpt.py"
    # Use text=True and capture_output=False to ensure logs are streamed to console
    subprocess.run(cmd, shell=True, check=True, text=True, capture_output=False)
    # Ensure logs are committed to the volume
    logs_volume.commit()


@app.local_entrypoint()
def main():
    """
    run with:
    modal run run_modal.py
    """
    # first, ensure the dataset is ready
    prepare_dataset.remote()
    # then, run the training
    fit.remote()
