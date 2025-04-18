import torch
import torch.nn as nn
import torch.nn.functional as functional

class SignatureEmbeddingCreator(nn.Module):
    def __init__(self):
        # the feature extractor learns and captures meaningful details from \
        # the raw input, while the final embedding layer summarizes this \
        # information into a compact representation suitable for \
        # decision-making or further processing.

        super().__init__()
        # Convolutional blocks to learn feature representations from the input image.
        self.feature_extractor = nn.Sequential(
            # Block 1: From 3 channels to 64 feature maps
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            # Normalizes result from previous layer
            nn.BatchNorm2d(64),
            # Introduces non-linearity into the network
            nn.ReLU(),
            # Reduces dimension by half
            nn.MaxPool2d(kernel_size=2),  # Reduces spatial dimension by half

            # Block 2: From 64 to 128 feature maps
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: From 128 to 256 feature maps
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Fully connected layers to reduce the convolutional output to an embedding.
        # Adjust the flattened size (here: 256 * 16 * 16) according to your input image dimensions.
        self.final_embedding_layer = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # Final embedding with dimension 128.
        )

    def forward(self, input_images):
        # 1. Extract high‑level feature maps from the input images.
        #    `feature_maps` has shape [batch_size, channels, height, width].
        feature_maps = self.feature_extractor(input_images)
        # Flatten the tensor. Converts the multidimensional tensor into a single long vector per sample
        batch_size = feature_maps.size(0)
        flattened_maps = feature_maps.view(batch_size, -1)
        # creates a compact, lower dimensional representation
        embeddings = self.final_embedding_layer(flattened_maps)
        return embeddings

class SiameseNetworkEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize one branch of the network that will be shared.
        # Shared embedding model for generating signature embeddings
        self.signature_embedding_model = SignatureEmbeddingCreator()

    def forward(self, signature_image_a, signature_image_b):
        # Pass both inputs through the identical embedding network.
        embedding_a = self.signature_embedding_model(signature_image_a)
        embedding_b = self.signature_embedding_model(signature_image_b)
        return embedding_a, embedding_b

    def get_embedding(self, signature_image):
        # This helper method returns the embedding for a single input.
        return self.signature_embedding_model(signature_image)


# Example usage:
if __name__ == '__main__':
    # 1. Instantiate the Siamese network. network.
    model = SiameseNetworkEmbedding()

    # 2. Create two batches of “fake” signature images.
    #    Here we use random noise just to test dimensions:
    #    – batch_size = 8
    #    – channels  = 3 (e.g. RGB)
    #    – height   = 128
    #    – width    = 128
    input1 = torch.randn(8, 1, 128, 128)
    input2 = torch.randn(8, 1, 128, 128)

    # 3. Forward both batches through the shared embedding model.
    #    The model call invokes its __call__, which in turn \
    #    runs forward(input1, input2).
    embed1, embed2 = model(input1, input2)

    # 4. Print out the shapes of the returned embeddings.
    #    You should see something like:
    #      Embedding shapes: torch.Size([8, 128]) torch.Size([8, 128])
    #    confirming that for each of the 8 samples you got a 128‑dim vector.
    print("Embedding shapes:", embed1.shape, embed2.shape)
