class ImageRegistration:
    """
    A class for image registration using various methods.
    
    This class handles loading images, performing normalization,
    and registering images using different methods, including:
    - Rigid registration (phase cross correlation)
    - Non-rigid registration (optical flow)
    - Feature-based registration (SIFT, ORB)
    """
    
    def __init__(self, crop_size=0):
        """
        Initialize the ImageRegistration class.
        
        Parameters:
        -----------
        crop_size : int
            Size of the border to crop from the images (default: 0)
        """
        self.crop_size = crop_size
        self.base_image = None
        self.target_image = None
        self.base_image_normalized = None
        self.target_image_normalized = None
        self.registered_image = None
        self.flow_vectors = None  # (u, v) for optical flow
        self.flow_magnitude = None
        self.shift = None  # For rigid registration
        
        # For visualization
        self.rgb_before = None
        self.rgb_after = None
        
    def load_and_crop_images(self, base_path, target_path):
        """
        Load images from file paths and crop borders if needed.
        
        Parameters:
        -----------
        base_path : str
            Path to the base image file
        target_path : str
            Path to the target image file
        
        Returns:
        --------
        tuple
            (base_image, target_image)
        """
        from skimage import io
        import h5py
        import numpy as np
        import os
        
        # Determine file extension
        _, ext = os.path.splitext(base_path)
        ext = ext.lower()
        
        # Load base image based on file extension
        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.base_image = io.imread(base_path)
        elif ext == '.h5':
            with h5py.File(base_path, 'r') as f:
                # Assuming the image is stored in the first dataset
                dataset_name = list(f.keys())[0]
                self.base_image = np.array(f[dataset_name])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        # Load target image
        _, ext = os.path.splitext(target_path)
        ext = ext.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            self.target_image = io.imread(target_path)
        elif ext == '.h5':
            with h5py.File(target_path, 'r') as f:
                # Assuming the image is stored in the first dataset
                dataset_name = list(f.keys())[0]
                self.target_image = np.array(f[dataset_name])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Convert to grayscale if needed
        if len(self.base_image.shape) > 2 and self.base_image.shape[2] >= 3:
            from skimage import color
            # If RGBA, convert RGB part to grayscale
            if self.base_image.shape[2] >= 3:
                self.base_image = color.rgb2gray(self.base_image[:, :, :3])
        
        if len(self.target_image.shape) > 2 and self.target_image.shape[2] >= 3:
            from skimage import color
            if self.target_image.shape[2] >= 3:
                self.target_image = color.rgb2gray(self.target_image[:, :, :3])
        
        # Crop images if crop_size > 0
        if self.crop_size > 0:
            self.base_image = self._crop_border(self.base_image, self.crop_size)
            self.target_image = self._crop_border(self.target_image, self.crop_size)
        print (self.base_image)    
        return self.base_image, self.target_image
    
    def normalize_images(self):
        """
        Normalize both images to have the same minimum and maximum values.
        
        Returns:
        --------
        tuple
            (normalized_base_image, normalized_target_image)
        """
        import numpy as np
        
        # Make copies to avoid modifying the originals
        self.base_image_normalized = self.base_image.copy()
        self.target_image_normalized = self.target_image.copy()
        
        # Normalize base image
        base_min = np.min(self.base_image_normalized)
        base_max = np.max(self.base_image_normalized)
        self.base_image_normalized = (self.base_image_normalized - base_min) / (base_max - base_min)
        
        # Normalize target image
        target_min = np.min(self.target_image_normalized)
        target_max = np.max(self.target_image_normalized)
        self.target_image_normalized = (self.target_image_normalized - target_min) / (target_max - target_min)
        
        return self.base_image_normalized, self.target_image_normalized
    
    def register_images(self, method='rigid', threshold=10):
        """
        Register the target image with the base image using the specified method.
        
        Parameters:
        -----------
        method : str
            Registration method ('rigid', 'nonrigid', 'sift', 'orb')
        threshold : int
            Threshold for optical flow vectors (used in nonrigid registration)
            
        Returns:
        --------
        registered_image : ndarray
            The registered target image
        """
        if method == 'rigid':
            return self._rigid_registration()
        elif method == 'nonrigid':
            return self._nonrigid_registration(threshold)
        elif method == 'sift':
            return self._feature_based_registration('sift')
        elif method == 'orb':
            return self._feature_based_registration('orb')
        else:
            raise ValueError(f"Unsupported registration method: {method}")
    
    def create_visualization(self, method='rigid'):
        """
        Create visualization images: RGB overlays before and after registration.
        
        Parameters:
        -----------
        method : str
            Registration method ('rigid', 'nonrigid', 'sift', 'orb')
            
        Returns:
        --------
        tuple
            (rgb_before, rgb_after)
        """
        import numpy as np
        
        # Make sure all images are normalized for proper RGB composition
        if self.base_image_normalized is None or self.target_image_normalized is None:
            self.normalize_images()
            
        # Create RGB visualization before registration
        self.rgb_before = np.zeros((self.base_image_normalized.shape[0], 
                                    self.base_image_normalized.shape[1], 3))
        self.rgb_before[..., 0] = self.target_image_normalized  # Red channel
        self.rgb_before[..., 1] = self.base_image_normalized    # Green channel
        self.rgb_before[..., 2] = self.base_image_normalized    # Blue channel
        
        # Create RGB visualization after registration
        if self.registered_image is not None:
            # Normalize registered image for visualization
            reg_min = np.min(self.registered_image)
            reg_max = np.max(self.registered_image)
            registered_normalized = (self.registered_image - reg_min) / (reg_max - reg_min)
            
            self.rgb_after = np.zeros_like(self.rgb_before)
            self.rgb_after[..., 0] = registered_normalized       # Red channel
            self.rgb_after[..., 1] = self.base_image_normalized  # Green channel
            self.rgb_after[..., 2] = self.base_image_normalized  # Blue channel
        
        return self.rgb_before, self.rgb_after
    
    def generate_quiver_plot_data(self, step=20):
        """
        Generate data for creating a quiver plot of the flow vectors.
        
        Parameters:
        -----------
        step : int
            Sampling step for the flow vectors (higher values mean fewer arrows)
            
        Returns:
        --------
        dict
            Dictionary containing x, y, u, v arrays for the quiver plot
        """
        import numpy as np
        
        if self.flow_vectors is None:
            return None
            
        u, v = self.flow_vectors
        nl, nc = self.base_image_normalized.shape
        
        # Create meshgrid for arrow positions
        y, x = np.mgrid[:nl:step, :nc:step]
        
        # Subsample flow vectors
        u_sub = u[::step, ::step]
        v_sub = v[::step, ::step]
        
        return {
            'x': x,
            'y': y,
            'u': u_sub,
            'v': v_sub
        }
    
    def calculate_histogram_data(self):
        """
        Calculate histogram data for the flow vectors.
        
        Returns:
        --------
        dict
            Dictionary containing histogram data
        """
        import numpy as np
        
        if self.flow_vectors is None or self.flow_magnitude is None:
            return None
            
        u, v = self.flow_vectors
        
        # Flatten arrays for histogram
        norm_flat = self.flow_magnitude.flatten()
        u_flat = np.abs(u.flatten())
        v_flat = np.abs(v.flatten())
        
        return {
            'norm': norm_flat,
            'u': u_flat,
            'v': v_flat
        }
    
    def _crop_border(self, img, border_size):
        """
        Crop the borders of an image.
        
        Parameters:
        -----------
        img : ndarray
            Input image
        border_size : int
            Size of the border to crop
            
        Returns:
        --------
        ndarray
            Cropped image
        """
        if img.ndim == 2:  # Grayscale image
            y, x = img.shape
            return img[border_size:y-border_size, border_size:x-border_size]
        elif img.ndim == 3:  # RGB or multi-channel image
            y, x, _ = img.shape
            return img[border_size:y-border_size, border_size:x-border_size]
        else:
            raise ValueError("Unsupported image dimensions")
    
    def _rigid_registration(self):
        """
        Perform rigid registration using phase cross correlation.
        
        Returns:
        --------
        ndarray
            Registered target image
        """
        from skimage.registration import phase_cross_correlation
        from skimage import transform
        
        # Calculate shift between images
        self.shift, error, diffphase = phase_cross_correlation(
            self.base_image_normalized, 
            self.target_image_normalized
        )
        
        # Apply rigid transformation
        rigid_shift = transform.EuclideanTransform(
            translation=(-self.shift[1], -self.shift[0])
        )
        self.registered_image = transform.warp(
            self.target_image_normalized, 
            rigid_shift,
            mode='edge'
        )
        
        return self.registered_image
    
    def _nonrigid_registration(self, threshold=10):
        """
        Perform non-rigid registration using optical flow.
        
        Parameters:
        -----------
        threshold : int
            Threshold for optical flow vectors magnitude
            
        Returns:
        --------
        ndarray
            Registered target image
        """
        import numpy as np
        from skimage.registration import optical_flow_tvl1
        from skimage import transform
        
        # First do rigid registration to align globally
        self._rigid_registration()
        
        # Then perform optical flow for local distortions
        v, u = optical_flow_tvl1(
            self.base_image_normalized, 
            self.registered_image
        )
        
        # Store flow vectors and calculate magnitude
        self.flow_vectors = (u, v)
        self.flow_magnitude = np.sqrt(u**2 + v**2)
        
        # Apply threshold if specified
        if threshold > 0:
            mask = np.where(self.flow_magnitude > threshold, 1, 0)
            u = u * mask
            v = v * mask
            self.flow_vectors = (u, v)
        
        # Warp image using flow vectors
        nr, nc = self.base_image_normalized.shape
        row_coords, col_coords = np.meshgrid(
            np.arange(nr), 
            np.arange(nc), 
            indexing='ij'
        )
        
        warped_img = transform.warp(
            self.registered_image,
            np.array([row_coords + v, col_coords + u]), 
            mode='edge'
        )
        
        self.registered_image = warped_img
        return self.registered_image
    
    def _feature_based_registration(self, method='sift'):
        """
        Perform feature-based registration using SIFT or ORB.
        
        Parameters:
        -----------
        method : str
            Feature detector method ('sift' or 'orb')
            
        Returns:
        --------
        ndarray
            Registered target image
        """
        import numpy as np
        import cv2
        from skimage import transform
        
        # Convert images to uint8 format for OpenCV
        base_uint8 = (self.base_image_normalized * 255).astype(np.uint8)
        target_uint8 = (self.target_image_normalized * 255).astype(np.uint8)
        
        if method == 'sift':
            # Initialize SIFT detector
            sift = cv2.SIFT_create()
            
            # Find keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(base_uint8, None)
            kp2, des2 = sift.detectAndCompute(target_uint8, None)
            
        elif method == 'orb':
            # Initialize ORB detector
            orb = cv2.ORB_create()
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(base_uint8, None)
            kp2, des2 = orb.detectAndCompute(target_uint8, None)
        
        else:
            raise ValueError(f"Unsupported feature detection method: {method}")
            
        # Create matcher
        if method == 'sift':
            # FLANN parameters for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Store good matches using Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    
        else:  # ORB
            # Brute force matcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Sort by distance
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        # Extract location of good matches
        if len(good_matches) >= 4:  # Need at least 4 points for homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # Use homography to warp target image
            h, w = base_uint8.shape
            warped_img = cv2.warpPerspective(target_uint8, H, (w, h))
            
            self.registered_image = warped_img / 255.0  # Convert back to float range [0, 1]
            
        else:
            print(f"Not enough good matches found ({len(good_matches)})")
            # Fallback to rigid registration
            self._rigid_registration()
        
        return self.registered_image
