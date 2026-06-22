# Installation
Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

Create `robodiff` environment
```bash
mamba env create -f conda_environment.yaml
```
Open environment 
```bash
conda activate robodiff
```

Before starting downgrade `huggingface-hub`
```bash
pip install huggingface-hub==0.25.2
```

Change `train_diffusion_unet_real_hybrid_workspace.yaml` to have adequate `batch_size`. I switched to 1 to run locally, but it was originally 64. 

Currently, I'm not cropping the images further, they stay at 224 x 224. 