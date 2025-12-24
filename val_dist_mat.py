import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import os
import logging
import numpy as np

# Set up logging
log_file = "build_and_compute_distance_matrix.log"
log_file = os.path.join("logs", log_file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def build_and_compute_distance_matrix(
    eval_dir,
    batch_size=128,
    num_images=None,
    output_path="results_matrix.pt",
    norm=2
):

    ############################################################################
    # 1) LOAD DATASET + PREPARE LABELS
    ############################################################################
    logger.info("Building dataset...")

    dataset_eval = datasets.ImageFolder(
        root=eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    if num_images is not None:
        dataset_eval.samples = dataset_eval.samples[:num_images]

    N = len(dataset_eval)
    logger.info(f"Dataset has {N} images.")

    # Prepare loaders
    loader_A = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)
    loader_B = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False,
                          num_workers=8, pin_memory=True)

    ############################################################################
    # 2) PREALLOCATE RESULTS MATRIX  (N × N)
    ############################################################################
    # results[i, j] = distance
    results = torch.zeros((N, N), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ############################################################################
    # 3) MAIN LOOP FOR BATCH A
    ############################################################################
    start_A = 0
    with torch.no_grad():
        for batch_idx_A, (batch_A_cpu, labels_A) in enumerate(loader_A):
            logger.info(f"[A] Batch {batch_idx_A+1}/{len(loader_A)}")

            batch_A = batch_A_cpu.flatten(1).to(device)
            bszA = batch_A.size(0)
            end_A = start_A + bszA

            ########################################################################
            # 3A) A×A block (upper triangle only) — VECTORIZED WRITES
            ########################################################################

            # L2 distance per image
            dist_matrix = torch.cdist(batch_A, batch_A, p=norm)

            # Create matrix of global indices
            idx_i = torch.arange(start_A, end_A).unsqueeze(1).expand(bszA, bszA)
            idx_j = torch.arange(start_A, end_A).unsqueeze(0).expand(bszA, bszA)

            # Upper triangle mask (exclude diagonal)

            # Labels for broadcasting
            # Vectorized writes
            results[idx_i, idx_j] = dist_matrix.cpu()

            ########################################################################
            # 3B) A×B blocks for later batches B — VECTORIZED
            ########################################################################

            for batch_idx_B, (batch_B_cpu, labels_B) in enumerate(tqdm.tqdm(loader_B)):

                # Skip B <= A
                if batch_idx_B <= batch_idx_A:
                    continue

                start_B = batch_idx_B * batch_size

                batch_B = batch_B_cpu.flatten(1).to(device)
                bszB = batch_B.size(0)
                end_B = start_B + bszB

                # Compute distance block A×B
                dist_AB = torch.cdist(batch_A, batch_B, p=norm).cpu()

                # Global index grids
                idx_i = torch.arange(start_A, end_A).unsqueeze(1).expand(bszA, bszB)
                idx_j = torch.arange(start_B, end_B).unsqueeze(0).expand(bszA, bszB)

                # Write full rectangular block in one vector call
                results[idx_i, idx_j] = dist_AB
                results[idx_j.T, idx_i.T] = dist_AB.T # Symmetric
                assert torch.allclose(
                    torch.diag(results[start_A:end_A, start_A:end_A]),
                    torch.zeros(end_A - start_A),
                    atol=1e-6
                ), "Diagonal entries are not zero!"

            start_A = end_A
            

        ########################################################################
        # 4) SAVE RESULTS
        ########################################################################
        logger.info("Saving results...")
        torch.save(results, output_path)
        logger.info(f"Done: {output_path}")


if __name__ == '__main__':
    logger.info("Starting script...")

    norms = [1, np.inf]

    for norm in norms:
        logger.info(f"Computing distance matrix with p = {norm}")
        if type(norm) == int:
            norm_name = f"l{norm}"
        else:
            norm_name = "linf"
        build_and_compute_distance_matrix(
            eval_dir="/mnt/data/datasets/imagenet/val/",
            batch_size=25,
            num_images=None,
            norm=norm,
            output_path=f"results_imagenet_stats/val_dist_matrix_{norm_name}.pt"
        )