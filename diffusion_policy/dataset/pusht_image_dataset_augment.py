import logging
import os
import time

import numpy as np
import torch
import zarr
from transformers import AutoImageProcessor, AutoModel

from diffusion_policy.common.pylogging_util import setup_logging


def add_embeddings(
    hf_model_name="facebook/dinov2-base", store_key="dinov2_base", step=200
):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dist_group = zarr.open(
        os.path.expanduser("./data/pusht/pusht_cchi_v7_replay.zarr"), "a"
    )

    processor = AutoImageProcessor.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)
    model.to(device)

    start_time = time.perf_counter()
    for _, group in dist_group.items():
        for k1, v1 in group.items():
            if k1 == "img":
                embeddings = []
                for i in range(0, v1.shape[0], step):
                    frame = torch.from_numpy(v1[i:i+step])
                    frame = (frame * 255).type(torch.uint8)
                    inputs = processor(images=frame, return_tensors="pt")
                    inputs.to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        last_hidden_states = outputs[0]
                        logging.info(
                            f"frame_idx: {i} - {list(last_hidden_states.cpu().mean(dim=1).shape)}"
                        )
                        embeddings.append(last_hidden_states.cpu().mean(dim=1))
    embeddings_np = np.concatenate(embeddings, axis=0)
    dist_group[f"/data/{store_key}"] = embeddings_np
    end_time = time.perf_counter()
    logging.info(
        f"processed {hf_model_name} embeddings for {embeddings_np.shape[0]} frames in {end_time - start_time} seconds, stored {embeddings_np.shape} in data store under key: /data/{store_key}"
    )


def test():
    dist_group = zarr.open(
        os.path.expanduser("./data/pusht/pusht_cchi_v7_replay.zarr"), "r"
    )
    
    for key1, val1 in dist_group.items():
        for key2, val2 in val1.items():
            logging.info(f"{key1} - {key2} - {val2.shape}")


if __name__ == "__main__":
    setup_logging()
    add_embeddings(hf_model_name="facebook/dinov2-large", store_key="dinov2_large")
    add_embeddings(hf_model_name="facebook/dinov2-base", store_key="dinov2_base")
    # add_embeddings(hf_model_name="facebook/vc1-large", store_key="vc1_large")
    test()
