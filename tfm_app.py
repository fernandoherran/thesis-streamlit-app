# Import libraries
########################

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from aux_dependencies import SessionState
from aux_dependencies.app_functions import *

# Load neural network
########################

# Load model structure from json
json_file = open('aux_dependencies/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load model weights
model.load_weights('aux_dependencies/model_weights.h5')

# Define app pages
########################

### Home page ###

def home():
    
    st.title('Alzheimer detection')
    
    st.markdown("<div style='text-align: justify'><a href='https://www.nia.nih.gov/health/what-alzheimers-disease'>Alzheimer’s disease</a> is an irreversible brain disorder where neurons stop functioning, lose connection with other neurons and die. This disease slowly destroys memory and thinking skills, and is the most common cause of dementia among older adults. <br/><br/> </div>", unsafe_allow_html=True)  
            
    st.markdown("<div style='text-align: justify'>\nOne of the technices used to identify Alzheimer's disease in patients is the use of Magnetic Resonance Imaging (MRI) of the head. MRI can detect brain abnormalities. In the early stages of Alzheimer's disease, an MRI scan of the brain may be normal. However, in later stages, MRI may show a decrease in the size of different areas of the brain. <br/><br/> </div>", unsafe_allow_html=True) 
    
    st.markdown("<div style='text-align: center'> <img src='data:image/png;base64,{}' class='img-fluid' width='500' >".format(img_to_bytes('./aux_dependencies/alzheimer-healthy.png')), unsafe_allow_html=True,)
    
    st.markdown("<div style='text-align: justify'>\nThe importance of the use of Artificial Intelligence (AI) in the industry in general, and in medical applications in particular, is one of the biggest reasons to carry out this study. The great variety of neural networks architectures that can be applied to solve Deep Learning problems needs special attention, and knowing how to choose the correct one is an important factor to take into account. <br/><br/> </div>", unsafe_allow_html=True)  
    
    st.markdown("<div style='text-align: justify'>The aim of this study is to implement a Convolutional Neural Network (CNN) to predict if a patient has Alzheimer’s disease using MRIs of his brain. This CNN consists of a binary classification, where there are two possible categories: patient is cognitively normal (CN) or patient has signs of Alzheimer’s disease (AD). Below it is shown a summary of the performance of the CNN trained in the thesis and used in this application.<br/><br/> </div>", unsafe_allow_html=True)
                      
    st.markdown("<div style='text-align: center'> <img src='data:image/png;base64,{}' class='img-fluid' width='500' >".format(img_to_bytes('./aux_dependencies/test_cm.png')), unsafe_allow_html=True,)

    
### About page ###

def about():
    st.title('About')
    
    st.markdown("<div style='text-align: justify'>This application has been built as part of the thesis carried out by Fernando Herrán Albelda for the Master in Data Science for KSchool. <br/><br/> </div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: justify'>This frontend has been done using Streamlit, and deployed using Heroku. The only input needed to run the process is to upload a Magnetic Resonance Image (MRI) of the brain of the patient, which must be in NIfTI format (.nii or .nii.gz).</div>", unsafe_allow_html=True) 
    
    st.subheader('Resources')
    st.markdown(f'''
- [Linkedin](http://www.linkedin.com/in/fernando-herran-albelda/fr)
- [App repository](https://github.com/fernandoherran/thesis-streamlit-app)
- [Thesis repository](https://github.com/fernandoherran/master-thesis)''')
    
    
### App page ###

def application():
    st.title('Application')
    st.subheader('Upload your MRI')
    
    # Upload file
    mri_file = st.file_uploader('Please upload your MRI file in Nifti format (.nii) or zipped Nifti (.nii.gz) format.',
                                type = ['nii','gz'])
    
    # Process
    if mri_file is not None:
            
        # Check if file is correct
        if 'nii' not in mri_file.name:
            st.text('Please choose a Nifti file')

        if 'nii' in mri_file.name:
            
            # Save temporarily the uploaded file as nifti
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
                volume = np.reshape(volume, (1,) + volume.shape)
                st.text(f'Processing the MRI file. It could take around 30 seconds...')
                
                # Apply neural network to model
                prediction = model.predict(volume)
                
                # Get activation map
                conv_heatmap = get_activation_map(model, volume)
                
                st.subheader('Alzheimer detection')
                
                if prediction > 0.5:
                    card('Alzheimer detected', 
                         f'with a probability of {prediction[0][0] * 100:.2f} %')
                else:
                    card('Cognitively normal', 
                         f'with a probability of {prediction[0][0] * 100:.2f} %')
                
                session_state.checkboxed = True
                      
                # Show activaiton maps
                st.subheader('Show activation maps')
                act_map = st.selectbox('',
                                       ('','First view', 'Second view', 'Third view'))
                
                if act_map == 'First view':
                    st.text(f'Extracting the activation map. It could take around 30 seconds...')
                    
                    fig_plotly = act_map_2d(conv_heatmap, view = "first")
                    st.plotly_chart(fig_plotly)
        
                elif act_map == 'Second view':
                    st.text(f'Extracting the activation map. It could take some seconds...')
                    fig_plotly = act_map_2d(conv_heatmap, view = "second")
                    st.plotly_chart(fig_plotly)

                elif act_map == 'Third view':
                    st.text(f'Extracting the activation map. It could take some seconds...')
                    fig_plotly = act_map_2d(conv_heatmap, view = "third")
                    st.plotly_chart(fig_plotly)
                    
# Define main process
########################

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


# Run main
if __name__ == '__main__':
    main()
