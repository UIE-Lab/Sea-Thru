import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import scipy as sp
import math

def srgb_to_linear(srgb_image: np.ndarray) -> np.ndarray:
    """Converts an sRGB image (float, 0-1 range) to a linear RGB color space."""
    mask = srgb_image <= 0.04045
    linear_image = np.zeros_like(srgb_image, dtype=np.float32)
    linear_image[mask] = srgb_image[mask] / 12.92
    linear_image[~mask] = np.power((srgb_image[~mask] + 0.055) / 1.055, 2.4)
    return linear_image

def linear_to_srgb(linear_image: np.ndarray) -> np.ndarray:
    """Converts a linear RGB image (float, 0-1 range) back to the sRGB color space."""
    mask = linear_image <= 0.0031308
    srgb_image = np.zeros_like(linear_image, dtype=np.float32)
    srgb_image[mask] = linear_image[mask] * 12.92
    srgb_image[~mask] = 1.055 * np.power(linear_image[~mask], 1.0 / 2.4) - 0.055
    return srgb_image

def depth_map_color_constancy(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    sigma: float = 0.25,
    epsilon: float = 0.1,
    iterations: int = 100,
    convergence_threshold: float = 1e-6,
    f_factor: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:

    log_image = np.log(rgb_image + 1e-9)
    height, width = log_image.shape[:2]
    max_dimension = max(width, height)
    p = 1.0 / ((sigma * max_dimension)**2 + 1)
    
    log_illuminant_estimate = np.copy(log_image)
    neighborhood_avg = np.zeros_like(log_illuminant_estimate)
    pad_width = 1

    print(f"\nStarting illuminant estimation (max: {iterations} iterations)...")
    for i in range(iterations):
        print(f"  Processing iteration {i+1}/{iterations}...")
            
        prev_log_illuminant_estimate = log_illuminant_estimate.copy()
        padded_illuminant = cv2.copyMakeBorder(log_illuminant_estimate, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REPLICATE)
        padded_depth = cv2.copyMakeBorder(depth_map, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REPLICATE)

        for x in range(height):
            for y in range(width):
                px, py = x + pad_width, y + pad_width
                center_depth = padded_depth[px, py]
                neighbor_coords = [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]
                valid_neighbors = []
                for nx, ny in neighbor_coords:
                    neighbor_depth = padded_depth[nx, ny]
                    if np.abs(neighbor_depth - center_depth) <= epsilon:
                        valid_neighbors.append(padded_illuminant[nx, ny])
                
                if len(valid_neighbors) > 0:
                    neighborhood_avg[x, y] = np.mean(valid_neighbors, axis=0)
                else:
                    neighborhood_avg[x, y] = log_illuminant_estimate[x, y]

        log_illuminant_estimate = log_image * p + neighborhood_avg * (1 - p)

        diff = np.mean(np.abs(log_illuminant_estimate - prev_log_illuminant_estimate))
        if diff < convergence_threshold:
            print(f"Convergence reached at iteration {i+1}. Stopping.")
            break
            
    print("Iterations finished.")

    final_log_illuminant = log_illuminant_estimate + np.log(f_factor)
    illuminant_map_linear = np.exp(final_log_illuminant)
    
    reflectance_log = log_image - final_log_illuminant
    corrected_image_linear = np.exp(reflectance_log)

    return illuminant_map_linear, corrected_image_linear

def backscatter_estimation(img, depths, num_bins=10, fraction=0.01):
    """
    Estimates the backscatter from a LINEAR input image.
    """
    print("Step 1: Finding candidate points for backscatter estimation...")
    z_max, z_min = np.max(depths), np.min(depths)
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r, points_g, points_b = [], [], []
    
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths >= a, depths <= b))
        if locs[0].size == 0:
            continue
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        vals = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = vals[:math.ceil(fraction * len(vals))]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
        
    all_B_pts = [np.array(points_r), np.array(points_g), np.array(points_b)]
    channel_names = ['Red (R)', 'Green (G)', 'Blue (B)']

    print("\nStep 2: Starting the model fitting process for each RGB channel...")
    all_backscatter_curves = []
    
    def estimate_overestimate(d, B_inf, beta_B, J_prime, beta_D_prime):
        """Calculates the backscatter overestimate based on Equation 10 from the paper."""
        backscatter_term = B_inf * (1 - np.exp(-beta_B * d))
        residual_term = J_prime * np.exp(-beta_D_prime * d)
        return backscatter_term + residual_term

    def calculate_true_backscatter(d, B_inf, beta_B):
        """Calculates the main backscatter component from Equation 10."""
        return B_inf * (1 - np.exp(-1 * beta_B * d))

    bounds_lower = [0, 0, 0, 0]
    bounds_upper = [1, 5, 1, 5]

    for i, B_pts in enumerate(all_B_pts):
        print(f"-> Processing {channel_names[i]} channel...")
        if B_pts.shape[0] < 4:
            print("  Error: Model could not be fitted.", file=sys.stderr)
            all_backscatter_curves.append(np.zeros_like(depths))
            continue
        B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
        coefs = None
        best_loss = np.inf
        for _ in range(25):
            try:
                optp, _ = sp.optimize.curve_fit(
                    f=estimate_overestimate, xdata=B_depths, ydata=B_vals,
                    p0=np.random.random(4) * bounds_upper, bounds=(bounds_lower, bounds_upper))
                l = np.mean(np.abs(B_vals - estimate_overestimate(B_depths, *optp)))
                if l < best_loss:
                    best_loss = l
                    coefs = optp
            except RuntimeError:
                pass
        if coefs is not None:
            backscatter_curve = calculate_true_backscatter(depths, coefs[0], coefs[1])
            all_backscatter_curves.append(backscatter_curve)
        else:
            print("  Error: Model could not be fitted.", file=sys.stderr)
            all_backscatter_curves.append(np.zeros_like(depths))

    final_backscatter_image = np.stack(all_backscatter_curves, axis=-1)
    
    return final_backscatter_image

def coarse_estimation_beta_D(
    dc_linear: np.ndarray,
    depth_map: np.ndarray, 
    max_val: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the coarse estimation of beta_D(z) as per Sec. 4.4.2 of the paper.
    """
    # Step 1: Call the internal function 'depth_map_color_constancy' to estimate the illuminant map (Êc).
    illuminant_map_linear, corrected_image_linear = depth_map_color_constancy(
        rgb_image=dc_linear, 
        depth_map=depth_map
    )
    
    # Step 2: Apply Equation 12 from the paper.
    eps = 1e-9

    beta_D_coarse = np.minimum(max_val, -np.log(illuminant_map_linear + eps) / (depth_map[..., np.newaxis] + eps))   
    beta_D_map = np.maximum(0, beta_D_coarse)
    
    # Return both the coarse beta_D and the illuminant map, as it's needed for the next step.
    return beta_D_map, illuminant_map_linear

def refined_estimation_beta_D(
    beta_D_coarse: np.ndarray,
    depths: np.ndarray,
    illuminant_linear: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refines the coarse estimate of beta_D using the method from Sec. 4.4.3 of the paper.
    """
    eps = 1e-9
    
    def calculate_beta_D_model(z, a, b, c, d):
        """Equation 11 from the paper: Two-term exponential model."""
        return a * np.exp(b * z) + c * np.exp(d * z)

    def depth_residuals(params, z_data, illum_data):
        """Equation 17 from the paper: z - ẑ."""
        a, b, c, d = params
        beta_D_model = calculate_beta_D_model(z_data, a, b, c, d)
        
        # --- CHANGE 1: Reshape beta_D_model for broadcasting ---
        # (N,) -> (N, 1)
        beta_D_model_reshaped = beta_D_model[:, np.newaxis]
        
        z_reconstructed = -np.log(illum_data + eps) / (beta_D_model_reshaped + eps)
        
        # --- CHANGE 2: Reshape z_data for broadcasting and flatten the result ---
        # The error is calculated with shape (N, 3) and then flattened to (N*3,).
        error = z_data[:, np.newaxis] - z_reconstructed
        return error.flatten()

    # --- Optimization Preparation ---
    valid_pixels = np.where(
        (depths > eps) & 
        np.all(illuminant_linear > eps, axis=2) & 
        np.all(beta_D_coarse > eps, axis=2)
    )
    
    if len(valid_pixels[0]) < 100:
        print("Warning: Not enough valid pixels for refinement. Using coarse estimate.")
        return beta_D_coarse, None

    z_filtered = depths[valid_pixels]
    illum_filtered = illuminant_linear[valid_pixels]

    best_coefs = None
    best_loss = np.inf
    bounds = ([0, -np.inf, 0, -np.inf], [np.inf, 0, np.inf, 0])

    print("\nRefining beta_D by minimizing depth reconstruction error...")
    for i in range(25):
        try:
            initial_guess = np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.])
            result = sp.optimize.least_squares(
                fun=depth_residuals,
                x0=initial_guess,
                args=(z_filtered, illum_filtered),
                bounds=bounds,
                method='trf'
            )

            current_loss = result.cost
            if current_loss < best_loss:
                best_loss = current_loss
                best_coefs = result.x
                print(f"  Restart {i+1}/{25}: Found new best loss -> {best_loss:.4f}")

        except Exception as e:
            print(f"  Restart {i+1}/{25}: Optimization failed. {e}", file=sys.stderr)

    if best_coefs is None:
        print("Warning: Refinement failed. Using coarse estimate.")
        return beta_D_coarse, None

    print("Refinement successful. Generating final refined beta_D map.")
    refined_beta_D_map = calculate_beta_D_model(depths[..., np.newaxis], *best_coefs)
    
    return refined_beta_D_map, best_coefs

def main():
    """
    Main function to run the batch processing.
    """
    # --- 1. SETUP YOUR DIRECTORIES HERE ---
    input_dir = '/home/itu/OzanDemir_504192220/Data/UIEBD/raw'      # Folder with PNG images
    depth_dir = '/home/itu/OzanDemir_504192220/Data/UIEBD/depth' # Folder with NPY depth maps
    output_dir = './output'     # Folder where results will be saved

    # --- Check if directories exist ---
    if not os.path.isdir(input_dir):
        sys.exit(f"Error: Input directory not found at '{input_dir}'")
    if not os.path.isdir(depth_dir):
        sys.exit(f"Error: Depth directory not found at '{depth_dir}'")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all png files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    print(f"Found {len(image_files)} PNG images to process.")

    for image_name in image_files:
        try:
            print(f"\n--- Processing: {image_name} ---")
            
            # --- 2. CONSTRUCT FILE PATHS ---
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(input_dir, image_name)
            depth_name = f"{base_name}_depth.npy"
            depth_path = os.path.join(depth_dir, depth_name)

            if not os.path.exists(depth_path):
                print(f"  Warning: Depth map '{depth_name}' not found. Skipping.")
                continue

            # --- 3. LOAD AND PREPROCESS DATA ---
            bgr_image = cv2.imread(image_path)
            rgb_image_uint8 = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            depth_map_inverse = np.load(depth_path)
            
            new_size = (320, 320)
            rgb_image_uint8 = cv2.resize(rgb_image_uint8, new_size, interpolation=cv2.INTER_AREA)
            depth_map_inverse = cv2.resize(depth_map_inverse, new_size, interpolation=cv2.INTER_LINEAR)
            
            depth_inverted = depth_map_inverse.max() - depth_map_inverse
            min_val, max_val = depth_inverted.min(), depth_inverted.max()
            normalized_zero_one = (depth_inverted - min_val) / (max_val - min_val + 1e-6)
            d_final = (0.1 + normalized_zero_one * 0.9).astype(np.float32)

            # --- 4. RUN THE ALGORITHM ---
            Ic_srgb = rgb_image_uint8.astype(np.float32) / 255.0
            Ic_linear = srgb_to_linear(Ic_srgb)
            
            Bc_linear = backscatter_estimation(Ic_linear, d_final, fraction=0.02)
            Dc_linear = np.clip(Ic_linear - Bc_linear, 0, 1)
            beta_D_coarse, illuminant_map_linear = coarse_estimation_beta_D(Dc_linear, d_final)
            beta_D_refined, _ = refined_estimation_beta_D(
                beta_D_coarse=beta_D_coarse,
                depths=d_final,
                illuminant_linear=illuminant_map_linear
            )
            Jc_linear = Dc_linear / (np.exp(-beta_D_refined * d_final[..., np.newaxis]) + 1e-9)

            # --- 5. CONVERT RESULTS TO SAVABLE FORMAT ---
            print("  Converting results for saving...")
            Bc_srgb = np.clip(linear_to_srgb(Bc_linear), 0, 1)
            Dc_srgb = np.clip(linear_to_srgb(Dc_linear), 0, 1)
            illuminant_srgb = np.clip(linear_to_srgb(illuminant_map_linear), 0, 1)
            Jc_srgb = np.clip(linear_to_srgb(Jc_linear), 0, 1)

            # Dictionary of images to save
            images_to_save = {
                '01_Input_sRGB': Ic_srgb,
                '02_Backscatter_sRGB': Bc_srgb,
                '03_DirectTransmission_sRGB': Dc_srgb,
                '04_Illuminant_sRGB': illuminant_srgb,
                '05_Final_Jc_sRGB': Jc_srgb,
            }

            # --- 6. SAVE THE OUTPUT IMAGES ---
            for name, img_float in images_to_save.items():
                # Convert float [0, 1] to uint8 [0, 255]
                img_uint8 = (img_float * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                # Define output path and save
                output_path = os.path.join(output_dir, f"{base_name}_{name}.png")
                cv2.imwrite(output_path, img_bgr)
            
            print(f"  Successfully saved all results for {image_name}.")
        
        except Exception as e:
            print(f"  An unexpected error occurred while processing {image_name}: {e}")

    print("\nBatch processing complete.")


if __name__ == '__main__':
    main()