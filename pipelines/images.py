from modal import Image

training_image_conda = (
    Image.conda()
    .conda_install(
        "pytorch::pytorch",
        "torchaudio",
        "pandas",
        channels=["conda-forge"]
    )
)

training_image_pip = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        "torch==2.0.0",
        "torchaudio==2.0.0",
        "pandas",
        "tqdm",
        "wandb",
    )
)