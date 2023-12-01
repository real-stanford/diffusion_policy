expected_version = {
    "Python":       "3.8.17",
    
    "numpy":        "1.21.1",
    "scipy":        "1.10.1",
    
    "torch":        "2.1.0+cu118",
    "torchvision":  "0.16.0+cu121",
    "torchaudio":   "2.1.0+cu118",
    
    "gym":          "0.21.0",
    # "rl_games":     "1.1.4",
    "wandb":        "0.15.12",
    
    "zarr":        "2.16.1",
    "tqdm":        "4.66.1",
    "hydra":        "1.3.2",
    
    "click":        "8.1.7",
    "cv2":          "4.8.1",
}

def checkVersion(package_name, expected_version):
    import importlib
    package = importlib.import_module(package_name)
    print("✓" if package.__version__ == expected_version else "✗", package_name, package.__version__)


import sys
print("✓" if sys.version.split(" ")[0] == expected_version["Python"] else "✗", "Python", sys.version)

for package_name, expected_version in expected_version.items():
    if package_name == "Python":
        continue
    checkVersion(package_name, expected_version)
