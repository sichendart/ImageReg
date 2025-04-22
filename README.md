# Image Registration Web Application

This web application provides an interface for image registration using various methods including rigid registration, non-rigid registration, and feature-based registration (SIFT, ORB).

## Features

- Upload and register image pairs in multiple formats (JPG, PNG, TIFF, H5)
- Multiple registration methods:
  - Rigid registration (phase cross correlation)
  - Non-rigid registration (optical flow)
  - Feature-based registration (SIFT, ORB)
- Visualization of registration results:
  - RGB overlays before and after registration
  - Quiver plots showing flow vectors for non-rigid registration
  - Histograms of flow vector magnitudes
- Border cropping option
- Adjustable threshold for optical flow vectors

## Project Structure

```
image-registration-app/
├── app.py                  # Flask application
├── image_registration.py   # Image registration class
├── templates/
│   └── index.html          # Web interface template
├── uploads/                # Temporary storage for uploaded images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Requirements

- Python 3.8+
- Flask
- NumPy
- Matplotlib
- scikit-image
- OpenCV
- h5py (for H5 file support)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/image-registration-app.git
   cd image-registration-app
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your web browser and go to `http://127.0.0.1:5000`

## How to Use

1. Upload a base image (reference) and a target image (to be registered)
2. Select registration method:
   - **Rigid Registration**: Simple translation registration using phase cross correlation
   - **Non-Rigid Registration**: Advanced registration using optical flow to handle local distortions
   - **SIFT Feature-based**: Registration using Scale-Invariant Feature Transform
   - **ORB Feature-based**: Registration using Oriented FAST and Rotated BRIEF features
3. Optional: Set crop border size if needed
4. If using Non-Rigid Registration, you can set a threshold for the flow vectors
5. Click "Register Images" to process
6. View the results including:
   - Original images
   - RGB overlays before and after registration
   - Registered image
   - Flow vector field (for non-rigid registration)
   - Histograms (for non-rigid registration)

## Dependencies

Create a `requirements.txt` file with the following content:

```
Flask==2.3.2
numpy==1.24.3
matplotlib==3.7.1
scikit-image==0.20.0
opencv-python==4.7.0.72
h5py==3.8.0
Werkzeug==2.3.4
```
