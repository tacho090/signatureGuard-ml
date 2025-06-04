import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import os
from PIL import Image
from torchvision import transforms
import plotly.express as px
import plotly.io as pio

# Set template to a dark theme for better visualization
pio.templates.default = "plotly_dark"

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

def load_signatures_with_filenames(genuine_dir, forged_dir=None):
    """Load signature images from directories with their filenames for tooltips"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    signatures = []
    labels = []
    categories = []
    filenames = []
    pairs = []
    
    # Load genuine signatures (Person 1)
    genuine_files = sorted([f for f in os.listdir(genuine_dir) if f.lower().endswith('.png')])
    genuine_indices = []
    
    for i, file in enumerate(genuine_files):
        path = os.path.join(genuine_dir, file)
        img_tensor = load_and_transform_image(path, transform)
        signatures.append(img_tensor)
        labels.append('Original')
        categories.append(0)
        filenames.append(file)
        genuine_indices.append(len(signatures) - 1)
    
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
            labels.append('Forged')
            categories.append(1)
            filenames.append(file)
            forged_indices.append(len(signatures) - 1)
    
        # Create pairs between genuine and forged
        for i in range(min(5, len(genuine_indices))):
            for j in range(min(3, len(forged_indices))):
                pairs.append((genuine_indices[i], forged_indices[j]))
    
    return signatures, labels, categories, filenames, pairs

def find_challenging_forgeries(embeddings, categories, genuine_indices, forged_indices, num_connections=5):
    """Find forgeries that are closest to genuine signatures in the embedding space"""
    challenging_pairs = []
    
    # Calculate distances between all genuine and forged signatures
    distances = []
    for i in genuine_indices:
        for j in forged_indices:
            # Calculate Euclidean distance between embeddings
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append((i, j, dist))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[2])
    
    # Take the top N closest pairs (most challenging forgeries)
    for i, (gen_idx, forge_idx, dist) in enumerate(distances[:num_connections]):
        challenging_pairs.append((gen_idx, forge_idx))
        print(f"Challenging forgery {i+1}: Distance = {dist:.4f}")
    
    return challenging_pairs

def create_interactive_3d_visualization(embeddings, labels, categories, filenames, pairs=None):
    """Create an interactive 3D visualization using Plotly"""
    # Use t-SNE to reduce dimensionality to 3D
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_3d = tsne.fit_transform(embeddings)
    
    # Create a custom color scale
    color_scale = px.colors.qualitative.Bold
    
    # Create a figure
    fig = go.Figure()
    
    # Dictionary to map category indices to names for the legend
    category_names = {
        0: "Original Signatures",
        1: "Forged Signatures"
    }
    
    # Plot each category with different colors
    unique_categories = np.unique(categories)
    
    # Add scatter points for each category
    for cat in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_name = category_names.get(cat, f"Category {cat}")
        
        # Create hover text with filename and label information
        hover_texts = [f"File: {filenames[i]}<br>Type: {labels[i]}" for i in idx]
        
        # Add 3D scatter plot for this category
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[idx, 0],
            y=embeddings_3d[idx, 1],
            z=embeddings_3d[idx, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_scale[cat % len(color_scale)],
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=hover_texts,
            hoverinfo='text',
            name=cat_name
        ))
    
    # Find genuine and forged indices
    genuine_indices = [i for i, cat in enumerate(categories) if cat == 0]
    forged_indices = [i for i, cat in enumerate(categories) if cat == 1]
    
    # Find challenging forgeries (closest to genuine signatures)
    challenging_pairs = find_challenging_forgeries(
        embeddings, categories, genuine_indices, forged_indices, num_connections=10
    )
    
    # Add connections between genuine signatures and their challenging forgeries
    if challenging_pairs:
        print(f"Adding connections to {len(challenging_pairs)} challenging forgeries...")
        
        for pair_idx, (gen_idx, forge_idx) in enumerate(challenging_pairs):
            # Get the 3D coordinates
            x0, y0, z0 = embeddings_3d[gen_idx]
            x1, y1, z1 = embeddings_3d[forge_idx]
            
            # Create a line between the genuine signature and challenging forgery
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(color='white', width=0.5),
                opacity=0.5,
                showlegend=False,
                hoverinfo='text',
                text=f"Distance: {np.linalg.norm(embeddings[gen_idx] - embeddings[forge_idx]):.4f}"
            ))
    
    # Add labels for categories
    for cat in unique_categories:
        idx = [i for i, c in enumerate(categories) if c == cat]
        # Calculate the center of this category
        x_center = np.mean(embeddings_3d[idx, 0])
        y_center = np.mean(embeddings_3d[idx, 1])
        z_center = np.mean(embeddings_3d[idx, 2])
        
        cat_name = category_names.get(cat, f"Category {cat}")
        
        # Add text annotation
        fig.add_trace(go.Scatter3d(
            x=[x_center],
            y=[y_center],
            z=[z_center],
            mode='markers+text',
            marker=dict(
                size=0,
                opacity=0
            ),
            text=[cat_name],
            textposition="top center",
            textfont=dict(
                size=14,
                color=color_scale[cat % len(color_scale)]
            ),
            showlegend=False
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title="Interactive 3D Signature Embeddings",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            itemsizing="constant"
        )
    )
    
    # Save as interactive HTML
    output_path = "output_html/signature_embeddings_interactive.html"
    fig.write_html(output_path)
    print(f"Interactive visualization saved to {output_path}")
    
    return fig

def main():
    # Paths
    model_path = "trained_model/signature_siamese_state_v1.pth"
    genuine_dir = "dataset_for_training/person1/original"
    forged_dir = "dataset_for_training/person1/forged"
    
    # Load the model
    print("Loading model...")
    model, device = load_model(model_path)
    
    # Load signature images with filenames
    print("Loading signature images...")
    signatures, labels, categories, filenames, _ = load_signatures_with_filenames(
        genuine_dir, forged_dir
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    for sig in signatures:
        emb = get_embedding(model, sig, device)
        embeddings.append(emb.flatten())
    embeddings = np.vstack(embeddings)
    
    # Create interactive visualization
    print("Creating interactive 3D visualization...")
    fig = create_interactive_3d_visualization(embeddings, labels, categories, filenames)
    
    print("Visualization complete! Open the HTML file in your browser to interact with it.")

if __name__ == "__main__":
    main()
