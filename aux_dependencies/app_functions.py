# Import libraries
import numpy as np
import nibabel as nib
from scipy import ndimage
from aux_dependencies.deepbrain_package.extractor import Extractor

# Import tensorflow packahes
import tensorflow as tf
#import tensorflow.keras.backend as k
#from tensorflow.keras.models import load_model 
#from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras.metrics import BinaryAccuracy

# Activation maps packages
import cv2
import plotly.graph_objects as go
from skimage.transform import resize


def read_nifti_file(file):
    """
    Read and load nifti file.
    """
    
    # Read file
    volume = nib.load(file)

    # Get raw data
    volume = volume.get_fdata()
    
    # Exchange axis 0 and 2
    if volume.shape[1] == volume.shape[2]:
        print(f"{file} has a shape incompatible")
    
    return volume


def remove_skull(volume):
    """
    Extract only brain mass from volume.
    """
    
    # Initialize brain tissue extractor
    ext = Extractor()

    # Calculate probability of being brain tissue
    prob = ext.run(volume) 

    # Extract mask with probability higher than 0.5
    mask = prob > 0.5
    
    # Detect only pixels with brain mass
    volume [mask == False] = 0
    volume = volume.astype("float32")
    
    return volume


def normalize(volume):
    """
    Normalize the volume intensity.
    """
    
    I_min = np.amin(volume)
    I_max = np.amax(volume)
    new_min = 0.0
    new_max = 1.0
    
    volume_nor = (volume - I_min) * (new_max - new_min)/(I_max - I_min)  + new_min
    volume_nor = volume_nor.astype("float32")
    
    return volume_nor


def cut_volume(volume):
    """
    Cut size of 3D volume.
    """
    
    if volume.shape[0] == 256:
        volume_new = volume[20:220,30:,:]
    
    if volume.shape[0] == 192:
        volume_new = volume[20:180,20:180,:]
    
    return volume_new


def resize_volume(volume):
    """
    Resize across z-axis
    """
    
    # Set the desired depth
    desired_height = 180
    desired_width = 180
    desired_depth = 110
    
    # Get current depth
    current_height = volume.shape[0]
    current_width = volume.shape[1]
    current_depth = volume.shape[2]
    
    # Compute depth factor
    height = current_height / desired_height
    width = current_width / desired_width
    depth = current_depth / desired_depth

    height_factor = 1 / height
    width_factor = 1 / width
    depth_factor = 1 / depth
    
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    
    # Resize across z-axis
    volume = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)
    
    return volume
    

def process_scan(file):
    """
    Read, skull stripping and resize Nifti file.
    """
    
    # Read Nifti file
    volume = read_nifti_file(file)
    
    # Extract skull from 3D volume
    volume = remove_skull(volume)
    
    # Cut 3D volume
    volume = cut_volume(volume)
    
    # Resize width, height and depth
    volume = resize_volume(volume)
    
    # Normalize pixel intensity
    volume = normalize(volume)
    
    return volume


def get_activation_map(model, volume, layer_name = 'conv3d_31'):

    # Layer to visualize
    layer_name = layer_name
    conv_layer = model.get_layer(layer_name)

    # Create a graph that outputs target convolution and output
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    # Index of the class
    class_idx = 0  # Model only returns one value from 0 to 1

    # Compute GRADIENT
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(volume)
        loss = predictions[:, class_idx]

    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    # Average gradients spatially
    weights = tf.reduce_mean(grads, axis=(0,1,2))

    # Build a ponderated map of filters according to gradients importance
    conv_layer_output = np.zeros(output.shape[0:3], dtype = np.float32)

    for index, weight in enumerate(weights):
        conv_layer_output += weight * output[:, :, :, index]

    # Reshape ponderated map of filters
    conv_layer_output = resize(conv_layer_output, (110, 130, 80))
    conv_layer_output = np.maximum(conv_layer_output, 0)
    conv_layer_output = (conv_layer_output - conv_layer_output.min()) / (conv_layer_output.max() - conv_layer_output.min())

    overlay_volume = cv2.addWeighted(volume[0,:,:,:], 0.6, conv_layer_output, 0.4, 0)
    
    return overlay_volume


def plotly_2d(volume): 
    
    min_value = np.amin(volume)
    max_value = np.amax(volume)
    
    # Define axis
    r = volume.shape[0]
    c = volume.shape[2]

    # Define frames
    nb_frames = volume.shape[1]
    
    # Define Plotly figure
    fig = go.Figure(frames = [go.Frame(data = go.Surface(z = (6.7 - k * 0.1) * np.ones((r, c)),
                                                         surfacecolor = np.flipud(volume[:,129 - k,:]),
                                                         cmin = min_value, 
                                                         cmax = max_value),
                                       name=str(k)) for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(z = 6.7 * np.ones((r, c)),
                             surfacecolor = np.flipud(volume[:,129,:]),
                             colorscale = 'jet',
                             cmin = min_value, 
                             cmax = max_value,
                             colorbar = dict(thickness=20, ticklen=4)))

    # Setup
    def frame_args(duration):
        return {"frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, 
                               "easing": "linear"}}

    sliders_setup = [{"pad": {"b": 60, "t": 60},
                      "len": 0.9,
                      "x": 0.1,
                      "y": 0,
                      "steps": [{"args": [[f.name], frame_args(0)],
                                 "label": str(k),
                                 "method": "animate",} for k, f in enumerate(fig.frames)]}]

    scene_setup = dict(xaxis_title = '',
                       yaxis_title = '',
                       zaxis_title = '',
                       xaxis = dict(showticklabels = False,
                                    showgrid = False,
                                    showbackground = False),
                       yaxis = dict(showticklabels = False,
                                    showgrid = False,
                                    showbackground = False),
                       zaxis = dict(showticklabels = False,
                                    showgrid = False,
                                    showbackground = False))

    # Layout
    fig.update_layout(width = 700,
                      height = 700,
                      scene = scene_setup,
                      scene_camera = dict(eye = dict(x = 0., y = 0., z = 1.3)),
                      dragmode = False,
                      sliders = sliders_setup)

    # Show figure
    #fig.show()
    return fig 


def plotly_3d(volume):
    
    def resize_volume(volume):
        """
        Resize across z-axis
        """

        # Set the desired depth
        desired_height = 55
        desired_width = 65
        desired_depth = 40

        # Get current depth
        current_height = volume.shape[0]
        current_width = volume.shape[1]
        current_depth = volume.shape[2]

        # Compute depth factor
        height = current_height / desired_height
        width = current_width / desired_width
        depth = current_depth / desired_depth

        height_factor = 1 / height
        width_factor = 1 / width
        depth_factor = 1 / depth

        # Rotate
        #img = ndimage.rotate(img, 90, reshape=False)

        # Resize across z-axis
        volume = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)

        return volume

    resized_volume = resize_volume(volume)
    
    X, Y, Z = np.mgrid[0:55:55j, 0:65:65j, 0:40:40j]

    fig = go.Figure(data = go.Volume(x = X.flatten(),
                                     y = Y.flatten(),
                                     z = Z.flatten(),
                                     value = resized_volume.flatten(),
                                     isomin = 0.2,
                                     isomax = 0.54,
                                     colorscale = "jet",
                                     opacity = 0.1,
                                     surface_count = 17))
    fig.show()