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

# Training image with CUDA support, was working on modal
train_image = (
    # "nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04", add_python="3.12"
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "numpy",
        "tqdm",
        "torch",
        "huggingface-hub",
    )
    .pip_install(
        "https://github.com/YouJiacheng/pytorch-nightly-whl-archive/releases/download/v2.7.0.dev20250208/torch-2.7.0.dev20250208+cu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        extra_index_url="https://download.pytorch.org/whl/nightly/cu126",
    )
    .workdir("/modded-nanogpt")
    .add_local_file("train_gpt.py", remote_path="/modded-nanogpt/train_gpt.py")
)

train_image_stable = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install(
        "numpy",
        "tqdm",
        "torch==2.7.0",
    )
    # have to install nccl manually because the stable pytorch specifies a version that is not compatible with cuda 12.6
    .pip_install(
        "nvidia-nccl-cu126==2.24.3",
        extra_options="--no-deps",
    )
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
    # image=train_image_stable,
    gpu="H100:8",
    volumes={
        "/modded-nanogpt/data/fineweb10B": volume,
        "/modded-nanogpt/logs": logs_volume,
    },
    # timeout after 60 minutes.
    # training should only take 3 minutes but warmup takes ~5-15 mins, etc.
    timeout=60 * 60,
)
def fit():
    from torch.distributed.run import parse_args, run

    args = [
        "--nproc-per-node=8",
        "train_gpt.py",
    ]
    # os.environ["NCCL_DEBUG"] = "INFO"
    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))

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
