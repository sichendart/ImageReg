from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import numpy as np
# force it to use a non-GUI backend; otherwise it raise an NSException issue (macOS + Matplotlib issue)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
from image_registration import ImageRegistration

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
# changed to 64 to handle image with large size
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'h5'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to generate image plot and convert to base64 for embedding in HTML
def plot_to_base64(img, title=None, cmap='gray'):
    buf = io.BytesIO()
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

# Helper function to generate quiver plot
def quiver_plot_to_base64(img, quiver_data, title=None):
    x = quiver_data['x']
    y = quiver_data['y']
    u = quiver_data['u']
    v = quiver_data['v']
    
    buf = io.BytesIO()
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, u, v, color='r', units='dots', 
               angles='xy', scale_units='xy', lw=1.5, scale=1)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

# Helper function to generate histogram plot
def histogram_plot_to_base64(hist_data, title=None):
    buf = io.BytesIO()
    plt.figure(figsize=(15, 5))
    
    # Create 3 subplots
    plt.subplot(1, 3, 1)
    plt.hist(hist_data['norm'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Magnitude')
    plt.xlabel('Pixel Displacement')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(hist_data['u'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.title('Horizontal')
    plt.xlabel('Pixel Displacement')
    
    plt.subplot(1, 3, 3)
    plt.hist(hist_data['v'], bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.title('Vertical')
    plt.xlabel('Pixel Displacement')
    
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_images():
    # Check if files are provided
    if 'baseImage' not in request.files or 'targetImage' not in request.files:
        return jsonify({'error': 'Both base and target images are required'}), 400
    
    base_file = request.files['baseImage']
    target_file = request.files['targetImage']
    
    # Check if filenames are valid
    if base_file.filename == '' or target_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(base_file.filename) or not allowed_file(target_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get registration method and crop size
    reg_method = request.form.get('registrationMethod', 'rigid')
    crop_size = int(request.form.get('cropSize', 0))
    threshold = int(request.form.get('threshold', 10))
    
    # Save uploaded files
    base_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(base_file.filename))
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(target_file.filename))
    
    base_file.save(base_path)
    target_file.save(target_path)
    
    try:
        # Initialize registration class
        reg = ImageRegistration(crop_size=crop_size)
        
        # Load and crop images
        reg.load_and_crop_images(base_path, target_path)
        
        # Normalize images
        reg.normalize_images()
        
        # Register images
        # reg.register_images(method=reg_method, threshold=threshold)
        reg.register_images(method=reg_method)

        # Create visualizations
        # reg.create_visualization(method=reg_method)
        reg.create_visualization()
        
        # Generate base64 images for display
        base_img_b64 = plot_to_base64(reg.base_image_normalized, title="Base Image")
        target_img_b64 = plot_to_base64(reg.target_image_normalized, title="Target Image")
        registered_img_b64 = plot_to_base64(reg.registered_image, title="Registered Image")
        rgb_before_b64 = plot_to_base64(reg.rgb_before, title="Before Registration (RGB overlay)", cmap=None)
        rgb_after_b64 = plot_to_base64(reg.rgb_after, title="After Registration (RGB overlay)", cmap=None)
        
        result = {
            'baseImage': base_img_b64,
            'targetImage': target_img_b64,
            'registeredImage': registered_img_b64,
            'rgbBefore': rgb_before_b64,
            'rgbAfter': rgb_after_b64
        }
        
        # Add quiver plot for non-rigid registration
        if reg_method == 'nonrigid':
            quiver_data = reg.generate_quiver_plot_data(nvec=20, threshold = threshold)
            if quiver_data:
                quiver_b64 = quiver_plot_to_base64(
                    reg.base_image_normalized, 
                    quiver_data, 
                    title="Flow Vector Field"
                )
                result['quiverPlot'] = quiver_b64
                
        # Add histogram data
        hist_data = reg.calculate_histogram_data()
        if hist_data:
            hist_b64 = histogram_plot_to_base64(
                hist_data, 
                title="Flow Vector Histograms"
            )
            result['histogramPlot'] = hist_b64
        
        return jsonify(result)
        
    except Exception as e:
        # to show where and why the error occur
        import traceback
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded files
        if os.path.exists(base_path):
            os.remove(base_path)
        if os.path.exists(target_path):
            os.remove(target_path)

if __name__ == '__main__':
    app.run(debug=True)
