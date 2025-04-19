import torch
from typing import Tuple
from siamese_model import SiameseNetworkEmbedding

class SiameseOnnxExporter:
    """
    Loads a Siamese PyTorch model checkpoint (.pth) and exports it to ONNX format.
    """
    def __init__(
        self,
        pth_path: str,
        onnx_path: str,
        input_shape: Tuple[int, int, int],
        opset_version: int = 12
    ):
        """
        Args:
            pth_path:        Path to the .pth checkpoint file of the Siamese model.
            onnx_path:       Destination path for the ONNX model file.
            input_shape:     A tuple (C, H, W) specifying input tensor dimensions.
            opset_version:   ONNX opset version to target.
        """
        self.pth_path = pth_path
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """
        Internal helper to instantiate the Siamese network and load saved weights.
        """
        model = SiameseNetworkEmbedding()
        state = torch.load(self.pth_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def export(self):
        """
        Exports the loaded Siamese model to ONNX, with two inputs and two outputs.
        """
        C, H, W = self.input_shape
        dummy1 = torch.randn(1, C, H, W, device=self.device)
        dummy2 = torch.randn(1, C, H, W, device=self.device)

        torch.onnx.export(
            self.model,
            (dummy1, dummy2),
            self.onnx_path,
            input_names=["signature_A", "signature_B"],
            output_names=["embedding_A", "embedding_B"],
            dynamic_axes={
                "signature_A": {0: "batch"},
                "signature_B": {0: "batch"},
                "embedding_A": {0: "batch"},
                "embedding_B": {0: "batch"},
            },
            opset_version=self.opset_version
        )
        print(f"Siamese ONNX model exported to {self.onnx_path}")

if __name__ == "__main__":
    # 1) Point to your trained .pth and where you want the ONNX to go
    exporter = SiameseOnnxExporter(
        pth_path="trained_model/signature_siamese_state_v1.pth",
        onnx_path="onnx/siamese.onnx",
        input_shape=(1, 128, 128),   # (channels, height, width)
        opset_version=12             # optional, defaults to 12
    )

    # 2) Perform the export
    exporter.export()
    # → creates “siamese.onnx” in your working directory