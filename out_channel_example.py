import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Define a simple 2D convolutional layer:
# - 3 input channels (e.g., an RGB image)
# - 64 output channels (64 filters)
# - Kernel size 3, stride 1, padding 1 to preserve spatial dimensions.
conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Create a dummy input tensor representing one RGB image of size 64x64.
input_tensor = torch.randn(1, 3, 64, 64)

# Pass the input tensor through the convolutional layer.
output_tensor = conv_layer(input_tensor)

# Extract one of the output channels, for example, channel index 10.
activation_map = output_tensor[0, 10, :, :].detach().cpu().numpy()

# Visualize the activation map as a grayscale image.
plt.figure(figsize=(4, 4))
plt.imshow(activation_map, cmap='gray')
plt.title("Activation Map (Channel 10)")
plt.axis('off')
plt.savefig("out_channel_example_map.png")
plt.show()
