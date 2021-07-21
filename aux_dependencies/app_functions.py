# Import libraries
########################

import streamlit as st
import numpy as np
import nibabel as nib
from scipy import ndimage
import base64
from aux_dependencies.deepbrain_package.extractor import Extractor

# Import tensorflow packahes
import tensorflow as tf

# Activation maps packages
import cv2
import plotly.graph_objects as go
from skimage.transform import resize

# Define functions 
########################

def img_to_bytes(img_path):
    '''
    Function used to read an image
    Input: image path
    ''' 
        
    with open(img_path, "rb") as image:
        f = image.read()
        img_bytes = bytearray(f)
           
    encoded = base64.b64encode(img_bytes).decode()
        
    return encoded


def card(header, body):
    '''
    Function used to show a card with text in Streamlit
    Inputs: header of the text, body of the text
    Output: card with the text displayed
    ''' 
    
    def card_begin_str(header):
    
        return (
            "<style>div.card{background-color:lightblue;border-radius: 5px;box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>"
            '<div class="card">'
            '<div class="container">'
            f"<h3 style='text-align: center;'>{header}</h3>"
        )
    
    def card_end_str():
        return "</div></div>"
    
    def html(body):
        st.markdown(body, unsafe_allow_html=True)

    lines = [card_begin_str(header), f"<p style='text-align: center;'>{body}</p>"]
    html("".join(lines))

def read_nifti_file(file):
    '''
    Function used to load and read a NIfTI file
    Inputs: NIfTI file directory
    Output: NIfTI image data
    ''' 
    
    # Load NIfTI file
    volume = nib.load(file)
    
    # Read image data from NIfTI file
    volume = volume.get_fdata()
        
    return volume


def remove_skull(volume):
    '''
    Function used to remove skull from brain image
    Inputs: MRI image
    Output: brain image without skull
    ''' 
    
    # Initialize brain mass extractor
    ext = Extractor()

    # Calculate probability of being brain mass
    prob = ext.run(volume) 

    # Extract mask with probability higher than 50% of being brain mass
    mask = prob > 0.5
    
    # Detect mask from image
    volume [mask == False] = 0
    volume = volume.astype('float32')
    
    return volume


def resize_volume(volume):
    '''
    Function used to resize the brain image
    Input: brain image
    Output: brain image resized
    '''
    
    # Exchange axis 0 and 2
    if volume.shape[1] == volume.shape[2]:
        volume = np.swapaxes(volume, 0, 2)

    # Cut volume
    if volume.shape[0] == 256:
        volume = volume[20:210, 40:240, 20:140]

    if volume.shape[0] == 192:
        volume = volume[25:175, 30:180,15:155]
    
    # Define desired shape
    input_shape = (110,130,80)
    
    # Compute factors
    height = volume.shape[0] / input_shape[0]
    width = volume.shape[1] / input_shape[1]
    depth = volume.shape[2] / input_shape[2]

    height_factor = 1 / height
    width_factor = 1 / width
    depth_factor = 1 / depth
    
    # Resize image
    volume_new = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)
        
    return volume_new
    

def normalize(volume):
    '''
    Function used to normalize the image pixel intensity
    Input: brain image
    Output: brain image normalized
    '''
    
    I_min = np.amin(volume)
    I_max = np.amax(volume)
    new_min = 0.0
    new_max = 1.0
    
    volume_nor = (volume - I_min) * (new_max - new_min)/(I_max - I_min)  + new_min
    volume_nor = volume_nor.astype('float32')
    
    return volume_nor


def process_scan(file):
    '''
    Function used to process a NIfTI file (read, remove skull, resize, normalize and save it as numpy file)
    Input: NIfTI file directory
    Output: directory where to save the NIfTI file processed as numpy’s compressed format (.npz)
    '''
    
    # Read Nifti file
    volume = read_nifti_file(file)
    
    # Remove skull from image
    volume = remove_skull(volume)
    
    # Resize 3D image
    volume = resize_volume(volume)
    
    # Normalize pixel intensity
    volume = normalize(volume)

    return volume


def get_activation_map(model, volume, layer_name = 'conv3d_23'):
    '''
    Function used to extract the activation maps of a CNN
    Inputs: CNN model, 3D image, last convolutional layer of the model
    Output: 3D image with the activation map
    ''' 

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


def act_map_2d(volume, view = "first"):  
    '''
    Function used to plot a 2D activation map
    Inputs: 3D image, view (first, second or third)
    Output: figure with the activation map
    ''' 
    
    def get_surface_frame(view, volume, k):
    
        if view == 'first':

            return volume[109 - k,:,:]

        elif view == 'second':

            return volume[:,129 - k,:]

        elif view == 'third':

            return volume[:,:,79-k]
    
    def get_surface_trace(view, volume):

        if view == 'first':

            return volume[109,:,:]

        elif view == 'second':

            return volume[:,129,:]

        elif view == 'third':

            return volume[:,:,79]
    
    if view == "first":
        
        # Define axis
        r = volume.shape[1]
        c = volume.shape[2]
        
        # Define frames
        nb_frames = volume.shape[0]
    
    elif view == 'second':
        
        # Define axis
        r = volume.shape[0]
        c = volume.shape[2]
        
        # Define frames
        nb_frames = volume.shape[1]
        
    elif view == 'third':
          
        # Define axis
        r = volume.shape[0]
        c = volume.shape[1]
        
        # Define frames
        nb_frames = volume.shape[2] 
  
    # Get minimum and maximum pixel values of the volume
    min_value = np.amin(volume)
    max_value = np.amax(volume)
    
    # Define Plotly figure
    fig = go.Figure(frames = [go.Frame(data = go.Surface(z = (6.7 - k * 0.1) * np.ones((r, c)),
                                                         surfacecolor = np.flipud(get_surface_frame(view, volume, k)),
                                                         cmin = min_value, 
                                                         cmax = max_value),
                                       name=str(k)) for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(z = 6.7 * np.ones((r, c)),
                             surfacecolor = np.flipud(get_surface_trace(view, volume)),
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


def act_map_3d(volume):
    '''
    Function used to plot a 3D activation map
    Inputs: 3D image
    Output: figure with the activation map
    ''' 
    
    ## Resize image (as to plot the original 3D image will take a high computational time)
    
    # Define desired shape
    input_shape = (55, 65, 40)

    # Compute factors
    height = volume.shape[0] / input_shape[0]
    width = volume.shape[1] / input_shape[1]
    depth = volume.shape[2] / input_shape[2]

    height_factor = 1 / height
    width_factor = 1 / width
    depth_factor = 1 / depth

    # Resize across z-axis
    resized_volume = ndimage.zoom(volume, (height_factor, width_factor, depth_factor), order=1)
        
    # Get minimum and maximum pixel values of the volume
    min_value = np.amin(resized_volume)
    max_value = np.amax(resized_volume)
    
    X, Y, Z = np.mgrid[0:55:55j, 0:65:65j, 0:40:40j]

    fig = go.Figure(data = go.Volume(x = X.flatten(),
                                     y = Y.flatten(),
                                     z = Z.flatten(),
                                     value = resized_volume.flatten(),
                                     isomin = min_value,
                                     isomax = max_value,
                                     colorscale = "jet",
                                     opacity = 0.1,
                                     surface_count = 17))
    fig.show()

