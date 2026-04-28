import numpy as np    

def create_patches(image, patch_size):
    """
    Given an image and patch_size, returns a list of patches.
    """
    patches = []
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            patch = image[i:i+ph, j:j+pw]
            patches.append(patch)
    
    return patches

def recompose(patches, img_height, img_width, patch_height, patch_width):
    """
    Recompose patches into full image.
    Args:
        patches (List[np.ndarray]): List of patches.
        img_height (int): Height of the full image.
        img_width (int): Width of the full image.
        patch_height (int): Height of the patches.
        patch_width (int): Width of the patches.
    Returns:
        np.ndarray: Full image.
    """
    num_rows = img_height // patch_height
    num_cols = img_width // patch_width
    full_img = np.zeros((img_height, img_width), dtype=patches[0].dtype)
    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * patch_width
            y = row * patch_height
            full_img[y:y+patch_height, x:x+patch_width] = patches[idx]
            idx += 1
    return full_img