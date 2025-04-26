import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import os
from PIL import Image
from torchvision import transforms
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Custom 3D arrow class for the visualization
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def load_model(model_path):
    """Load the trained Siamese model"""
    from siamese_model import SiameseNetworkEmbedding
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetworkEmbedding().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def load_and_transform_image(image_path, transform=None):
    """Load an image and apply transformations"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return transform(img).unsqueeze(0)  # Add batch dimension

def get_embedding(model, image_tensor, device):
    """Extract embedding from an image using the model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model.get_embedding(image_tensor)
    return embedding.cpu().numpy()

def load_signatures(genuine_dir, forged_dir=None, other_genuine_dir=None):
    """Load signature images from directories"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    signatures = []
    labels = []
    categories = []
    pairs = []  # To track which signatures should be connected with arrows
    
    # Load genuine signatures (Person 1)
    genuine_files = sorted([f for f in os.listdir(genuine_dir) if f.lower().endswith('.png')])
    genuine_indices = []
    
    for i, file in enumerate(genuine_files):
        path = os.path.join(genuine_dir, file)
        img_tensor = load_and_transform_image(path, transform)
        signatures.append(img_tensor)
        labels.append('GaV unf')  # Example label from the visualization
        categories.append(0)  # Category 0 for Person 1 genuine
        genuine_indices.append(len(signatures) - 1)  # Keep track of index
    
    # Create pairs between genuine signatures
    for i in range(len(genuine_indices)):
        for j in range(i + 1, min(i + 3, len(genuine_indices))):
            pairs.append((genuine_indices[i], genuine_indices[j]))
    
    # Load forged signatures if provided
    forged_indices = []
    if forged_dir:
        forged_files = sorted([f for f in os.listdir(forged_dir) if f.lower().endswith('.png')])
        for i, file in enumerate(forged_files):
            path = os.path.join(forged_dir, file)
            img_tensor = load_and_transform_image(path, transform)
            signatures.append(img_tensor)
            labels.append('Cc Ohg')  # Example label from the visualization
            categories.append(1)  # Category 1 for Person 1 forged
            forged_indices.append(len(signatures) - 1)  # Keep track of index
    
        # Create pairs between genuine and forged (first 5 pairs only)
        for i in range(min(5, len(genuine_indices))):
            for j in range(min(3, len(forged_indices))):
                pairs.append((genuine_indices[i], forged_indices[j]))
    
    # Load other person's genuine signatures if provided
    other_indices = []
    if other_genuine_dir:
        other_files = sorted([f for f in os.listdir(other_genuine_dir) if f.lower().endswith('.png')])
        for i, file in enumerate(other_files):
            path = os.path.join(other_genuine_dir, file)
            img_tensor = load_and_transform_image(path, transform)
            signatures.append(img_tensor)
            labels.append('O Dog')  # Example label from the visualization
            categories.append(2)  # Category 2 for Person 2 genuine
            other_indices.append(len(signatures) - 1)  # Keep track of index
        
        # Create pairs between other genuine and forged (first few pairs only)
        if forged_dir:
            for i in range(min(3, len(other_indices))):
                for j in range(min(2, len(forged_indices))):
                    pairs.append((other_indices[i], forged_indices[j]))
    
    return signatures, labels, categories, pairs

def visualize_embeddings_3d(embeddings, labels, categories, pairs=None):
    """Create a 3D visualization of signature embeddings"""
    # Use t-SNE to reduce dimensionality to 3D
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_3d = tsne.fit_transform(embeddings)
    
    # Create a colormap
    cmap = plt.cm.jet
    norm = plt.Normalize(0, max(categories))
    
    # Create a 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each category with different colors
    unique_categories = np.unique(categories)
    
    # Plot points
    for cat in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == cat]
        label = labels[idx[0]]
        scatter = ax.scatter(
            embeddings_3d[idx, 0], 
            embeddings_3d[idx, 1], 
            embeddings_3d[idx, 2],
            c=[cmap(norm(cat)) for _ in idx],
            label=label,
            alpha=0.7,
            s=50
        )
    
    # Add labels to the axes
    ax.set_xlabel('Dris bighn pace', fontsize=12)
    ax.set_ylabel('Embight Race', fontsize=12)
    ax.set_zlabel('Churr Dinwiz', fontsize=12)
    
    # Add grid in the background
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set view angle
    ax.view_init(elev=30, azim=45)
    
    # Add "boxes" and arrows similar to the example
    category_centers = {}
    for cat in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == cat]
        center = np.mean(embeddings_3d[idx], axis=0)
        category_centers[cat] = center
    
    # Add text boxes for labels
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
    
    # Add boxes and connection arrows
    for cat in unique_categories:
        if cat in category_centers:
            center = category_centers[cat]
            label = labels[[i for i, c in enumerate(categories) if c == cat][0]]
            
            # Add text box
            ax.text(center[0], center[1], center[2], label, 
                    ha='center', va='center', fontsize=10,
                    bbox=box_props, zorder=10)
    
    # Add connection arrows between related categories (example)
    if len(unique_categories) > 1 and 0 in category_centers and 1 in category_centers:
        arrow = Arrow3D([category_centers[0][0], category_centers[1][0]],
                        [category_centers[0][1], category_centers[1][1]],
                        [category_centers[0][2], category_centers[1][2]],
                        mutation_scale=20, lw=2, arrowstyle='->', color='gray')
        ax.add_artist(arrow)
    
    if len(unique_categories) > 2 and 2 in category_centers and 1 in category_centers:
        arrow = Arrow3D([category_centers[2][0], category_centers[1][0]],
                        [category_centers[2][1], category_centers[1][1]],
                        [category_centers[2][2], category_centers[1][2]],
                        mutation_scale=20, lw=2, arrowstyle='->', color='gray')
        ax.add_artist(arrow)
    
    # Add arrows connecting embedding pairs
    if pairs:
        print(f"Drawing arrows between {len(pairs)} embedding pairs...")
        # Only draw a subset of pairs if there are too many to avoid clutter
        max_arrows = min(50, len(pairs))
        for i, (idx1, idx2) in enumerate(pairs[:max_arrows]):
            # Get 3D coordinates for each pair
            p1 = embeddings_3d[idx1]
            p2 = embeddings_3d[idx2]
            
            # Calculate category for each point to determine arrow color
            cat1 = categories[idx1]
            cat2 = categories[idx2]
            
            # Set arrow color based on categories
            if cat1 == cat2:
                # Same category - use category color but lighter
                arrow_color = cmap(norm(cat1))
                alpha = 0.3
            else:
                # Different categories - use a gradient between the two colors
                arrow_color = 'gray'
                alpha = 0.4
            
            # Create and add arrow
            arrow = Arrow3D([p1[0], p2[0]],
                           [p1[1], p2[1]],
                           [p1[2], p2[2]],
                           mutation_scale=10, lw=1, 
                           arrowstyle='->', 
                           color=arrow_color,
                           alpha=alpha)
            ax.add_artist(arrow)
    
    # Add title
    plt.title('3D Signature Embedding Visualization', fontsize=16)
    
    # Add custom axis labels on the edges
    ax.text2D(0.05, 0.95, 'Ohktqur 12', transform=ax.transAxes, fontsize=12)
    ax.text2D(0.85, 0.95, 'Maun qür 13', transform=ax.transAxes, fontsize=12)
    
    # Maximize figure to fill screen
    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'window'):
        mng.window.state('zoomed')  # works on Windows
    
    plt.savefig('signature_embeddings_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Paths
    model_path = "trained_model/signature_siamese_state_v1.pth"
    genuine_dir = "dataset_for_training/person1/original"
    forged_dir = "dataset_for_training/person1/forged"
    
    # Check if we have person2 data
    other_genuine_dir = None
    if os.path.exists("dataset_for_training/person2/original"):
        other_genuine_dir = "dataset_for_training/person2/original"
    
    # Load the model
    print("Loading model...")
    model, device = load_model(model_path)
    
    # Load signature images
    print("Loading signature images...")
    signatures, labels, categories, pairs = load_signatures(genuine_dir, forged_dir, other_genuine_dir)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    for sig in signatures:
        emb = get_embedding(model, sig, device)
        embeddings.append(emb.flatten())  # Flatten if needed
    embeddings = np.vstack(embeddings)
    
    # Create visualization
    print("Creating 3D visualization...")
    visualize_embeddings_3d(embeddings, labels, categories, pairs)
    print("Visualization complete! Saved as 'signature_embeddings_3d.png'")

if __name__ == "__main__":
    main()
