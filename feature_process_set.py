import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from feature import generate_pipeline_features # Assuming your previous code is saved here
import cv2 

# --- Configuration ---
TRAIN_DIR = 'train'  # Root folder containing class subfolders (e.g., train/Apple, train/Banana)
OUTPUT_CSV = 'feature_data_matrix.csv'
N_COMPONENTS_PCA = 3 # For 3D visualization

def extract_features_from_dataset(root_dir):
    """
    Walks through the dataset, calls the feature extraction pipeline for 
    each image, and collects the feature vectors and labels.
    """
    all_features = []
    all_labels = []
    all_paths = []
    
    print(f"Starting feature extraction from directory: {root_dir}")
    
    # Iterate through all subdirectories (classes)
    for class_label in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_label)
        
        if os.path.isdir(class_path):
            print(f"--- Processing class: {class_label} ---")
            
            # Iterate through all files (images) in the class folder
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    
                    # Temporarily link the image file to 'img2.jpeg' 
                    temp_name = "img2.jpeg"
                    
                    try:
                        # 1. Create a link to the expected path
                        if os.path.exists(temp_name):
                            os.remove(temp_name)
                        os.link(image_path, temp_name)
                        
                        # 2. Call the feature extraction function
                        feature_vector = generate_pipeline_features() 
                        
                        # Ensure the vector is a 1D numpy array
                        if feature_vector is not None and feature_vector.ndim == 1:
                            all_features.append(feature_vector)
                            all_labels.append(class_label)
                            all_paths.append(image_path)
                            print(f"  Extracted features for: {image_file}")
                        else:
                            print(f"  Skipped (Feature vector invalid) for: {image_file}")

                    except Exception as e:
                        print(f"  ERROR processing {image_file}: {e}")
                    finally:
                        # 3. Clean up the temporary file
                        if os.path.exists(temp_name):
                            os.remove(temp_name)
    
    return np.array(all_features), all_labels, all_paths


def visualize_3d_pca(feature_matrix, labels, n_components=3):
    """
    Performs PCA on the feature matrix and displays a 3D scatter plot.
    """
    if feature_matrix.shape[0] < n_components:
        print("\nCannot run PCA: Too few samples compared to components.")
        return

    print("\nStarting 3D PCA visualization...")
    
    # 1. Handle NaN/Inf values (important for real-world data)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e10, neginf=-1e10)

    # 2. Perform PCA
    # Note: For better clustering results, it is often recommended to scale the features
    # before PCA, e.g., using sklearn.preprocessing.StandardScaler().
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(feature_matrix)
    pc_df = pd.DataFrame(data=principal_components, 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    
    # 3. Visualization setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    ax.set_zlabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2]*100:.2f}%)', fontsize=12)
    ax.set_title('3D PCA of Feature Vectors by Class', fontsize=15)
    
    
    targets = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('Spectral', len(targets)) # Use a colorful colormap
    
    # 4. Scatter Plot by Class
    for i, target in enumerate(targets):
        indices_to_keep = [j for j, label in enumerate(labels) if label == target]
        ax.scatter(pc_df.loc[indices_to_keep, 'PC1'],
                   pc_df.loc[indices_to_keep, 'PC2'],
                   pc_df.loc[indices_to_keep, 'PC3'],
                   c=colors(i),
                   label=target,
                   s=50)

    ax.legend(targets, loc='upper right', title="Classes")
    ax.grid()
    plt.show() # <-- ENABLED: This line displays the 3D plot.

# --- Execution ---
if __name__ == "__main__":
    # --- Step 1: Feature Extraction ---
    feature_matrix, labels, paths = extract_features_from_dataset(TRAIN_DIR)

    if feature_matrix.shape[0] == 0:
        print("\n⚠️ No features extracted. Check the 'train' directory structure and image files.")
    else:
        # --- Step 2: Prepare DataFrame ---
        n_features = feature_matrix.shape[1]
        header = (["v1_norm", "v2_norm", "Eigen_Ratio"] + 
                  [f"F{i+1}" for i in range(10)] + 
                  ["Global_ID", "Global_AR", "Local1_ID", "Local1_Ratio", "Local2_ID", "Local2_Ratio", "Mean_L", "Mean_a", "Mean_b"])
        
        df = pd.DataFrame(feature_matrix, columns=header)
        df.insert(0, 'Class_Label', labels)
        df.insert(1, 'Image_Path', paths)

        # --- Step 3: Save to CSV (Excel readable) ---
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Feature data saved successfully to {OUTPUT_CSV} (readable by Excel).")

        # --- Step 4: 3D PCA Visualization ---
        # This calls the function that performs PCA and displays the plot.
        visualize_3d_pca(feature_matrix, labels, n_components=N_COMPONENTS_PCA)