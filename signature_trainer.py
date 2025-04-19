import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from contrastive_loss import ContrastiveLoss
from data_loader import SignaturePairDataset
from siamese_model import SiameseNetworkEmbedding
from logger import Logger
import logging

log = Logger(
    name=__name__,
    level=logging.DEBUG,
    log_to_file=True,
    file_path="log/train.log"
).logger

class SignatureTrainer:
    def __init__(
        self,
        genuine_dir: str,
        forged_dir: str,
        batch_size: int = 32,
        margin: float = 1.0,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the SignatureTrainer with dataset paths, hyperparameters, and training components.

        Args:
            genuine_dir (str): Directory containing genuine signature images.
            forged_dir (str): Directory containing forged signature images.
            batch_size (int, optional): Number of pairs per batch. Defaults to 32.
            margin (float, optional): Margin for contrastive loss. Defaults to 1.0.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to GPU if available.
        """

        # 1. Dataset & DataLoader
        log.info("1. Load dataset")
        dataset = SignaturePairDataset(genuine_dir, forged_dir)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Model train
        log.info("2. Model Train")
        self.model = SiameseNetworkEmbedding().to(device)

        # 3. Contrastive Loss function
        log.info("3. Contrastive Loss Function")
        self.criterion = ContrastiveLoss(margin=margin)

        # 4. Optimizer - updates your model's weights during training
        log.info("4. Optimizer - update model's weights during training")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.device = device

    def train_epoch(self, epoch: int):
        """
        Run a single training epoch over the entire dataset.

        Args:
            epoch (int): Current epoch number, used for logging progress.
        """
        log.info("Starting training loop")
        self.model.train()
        total_loss = 0.0
        for batch_idx, (img1, img2, labels) in enumerate(self.dataloader, 1):
            log.debug("Processing images and labels")
            img1, img2, labels = (
                img1.to(self.device),
                img2.to(self.device),
                labels.to(self.device)
            )
            log.debug("Creating embeddings")
            emb1, emb2 = self.model(img1, img2)
            log.debug("Computing loss function")
            loss = self.criterion(emb1, emb2, labels)

            log.debug("Running algorithm optimizer")
            self.optimizer.zero_grad()
            log.debug("Performing backward loss function")
            loss.backward()
            log.debug("Processing step optimizer")
            self.optimizer.step()

            log.debug("Computing total loss")
            total_loss += loss.item()
        avg_loss = total_loss / len(self.dataloader)
        log.info(f"Epoch {epoch:02d}: Avg Loss = {avg_loss:.4f}")

    def run_training(self, epochs: int = 10):
        """
        Execute the training loop for the specified number of epochs.
        """
        for epoch in range(1, epochs + 1):
            log.info(f"********Training epoch {epoch}********")
            self.train_epoch(epoch)

class Main:
    @staticmethod
    def run():
        """
        Entry point for training: set up directories, hyperparameters, and start the trainer.
        """
        # adjust paths and hyperparameters as needed
        genuine_dir = "dataset_for_training/person1/original"
        forged_dir  = "dataset_for_training/person1/forged"
        trainer = SignatureTrainer(
            genuine_dir=genuine_dir,
            forged_dir=forged_dir,
            batch_size=16,
            margin=1.0,
            lr=1e-3
        )
        trainer.run_training(epochs=20)

        # Export the trained model
        # Save just the state_dict (recommended):
        log.info("Save model state_dict")
        torch.save(
            trainer.model.state_dict(),
            "trained_model/signature_siamese_state_v1.pth"
        )

        # Save the entire model object:
        log.info("Save full trainer model object")
        torch.save(
            trainer.model,
            "trained_model/signature_siamese_full_v1.pth"
        )

if __name__ == "__main__":
    Main.run()
