import numpy as np
import nibabel as nib
import tensorflow as tf
#from deepbrain import Extractor
from scipy import ndimage
from tensorflow.keras.models import load_model 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import BinaryAccuracy
#from skimage.transform import resize
import plotly.graph_objects as go


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

def resize_volume_2(volume):
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
    

def process_scan(file):
    """
    Read, skull stripping and resize Nifti file.
    """
    
    # Read Nifti file
    volume = read_nifti_file(file)
    
    # Extract skull from 3D volume
    # volume = remove_skull(volume)
    
    # Cut 3D volume
    #volume = cut_volume(volume)
    
    # Resize width, height and depth
    volume = resize_volume(volume)
    
    # Normalize pixel intensity
    volume = normalize(volume)
    
    return volume

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_cnn(model_name, path):

    #  Load model
    model = load_model(path + "Results/" + model_name + ".h5", 
                       custom_objects = {'f1': f1})

    # Define optimizer
    optimizer = Adam(learning_rate = 0.001, decay = 1e-6)

    # Compile model
    model.compile(loss = "binary_crossentropy",
                  optimizer = optimizer,
                  metrics = [BinaryAccuracy(), f1])

    return model