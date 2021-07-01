# Import libraries
import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from aux_dependencies import SessionState
from aux_dependencies.app_functions import *

## Load neural network

# Load model structure from json
json_file = open('aux_dependencies/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load model weights
model.load_weights('aux_dependencies/model_weights.h5')

## Define app pages

# Define Home page
def home():
    st.title('Alzheimer detection')
    st.text('The objective of this application is to...')

# Define About page
def about():
    st.title('About')
    st.subheader('About the App')
    st.text('The objective of this application is to...')
    st.subheader('Resources')
    st.markdown(f'''
- [Linkedin](http://www.linkedin.com/in/fernando-herran-albelda/fr)
- [App repository](https://www.github.com/fernandoherran)''')
    
# Define Application page
def application():
    st.title('Application')
    st.subheader('Upload your MRI')
    
    # Upload file
    mri_file = st.file_uploader('Upload your MRI file. App supports Nifti (.nii) and zipped Nifti (.nii.gz) files.',
                                type = ['nii','gz'])
    
    # Process
    if mri_file is not None:
            
        # Check if file is correct
        if 'nii' not in mri_file.name:
            st.text('Please choose a Nifti file')

        if 'nii' in mri_file.name:
            
            # Save uploaded file as nifti temporarily
            with open(mri_file.name, 'wb') as file: 
                newFileByteArray = bytearray(mri_file.read())
                file.write(newFileByteArray)
                
            session_state = SessionState.get(checkboxed = False)

            # Calculate prediction
            if st.button('Run process') or session_state.checkboxed:
                
                # Get MRI file path
                mri_file = "./" + str(mri_file.name)
                
                # Read and preprocess NiFTI file
                volume = process_scan(mri_file)
                
                # Reshape volume 
                volume = volume[10:120, 30:160, 15:95]
                volume = np.reshape(volume, (1,) + volume.shape)
                st.text(f'{volume.shape}')
                
                # Apply neural network to model
                prediction = model.predict(volume)
                
                # Get activation map
                conv_heatmap = get_activation_map(model, volume)
                
                st.subheader('Alzheimer detection')
                
                session_state.checkboxed = True
                
                # Show preprocessing
                if st.checkbox('Show preprocessing'):
                    st.text('skull stripping')
                    
                session_state.checkboxed = True
                      
                # Show class
                if st.checkbox('Show results'):
                    
                    if prediction > 0.5:
                        card('Alzheimer detected', 
                             f'with a probability of {prediction[0][0] * 100:.2f} %')
                    else:
                        card('Cognitively normal', 
                             f'with a probability of {prediction[0][0] * 100:.2f} %')
                        
                session_state.checkboxed = True
                
                # Show activation maps
                if st.checkbox('Show activation maps'):
                    fig_plotly = plotly_2d(conv_heatmap)
                    st.plotly_chart(fig_plotly)
                    
def card(header, body):
    
    def card_begin_str(header):
    
        return (
            "<style>div.card{background-color:lightblue;border-radius: 5px;box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);transition: 0.3s;}</style>"
            '<div class="card">'
            '<div class="container">'
            #f"<h3><b>style=text-align: center; color: red;'{header}</b></h3>"
            f"<h3 style='text-align: center;'>{header}</h3>"
        )
    
    def card_end_str():
        return "</div></div>"
    
    def html(body):
        st.markdown(body, unsafe_allow_html=True)

    lines = [card_begin_str(header), f"<p style='text-align: center;'>{body}</p>"]
    html("".join(lines))

## Define main
def main():
    
    # Define navigation panel
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Go to', 
                                 ['Home', 'Application', 'About'])
    
    # Define section pages
    if selection == 'Home':
        home()
    
    if selection == 'Application':
        application()
    
    if selection == 'About':
        about()


## Run main
if __name__ == '__main__':
    main()
