# Alzheimer’s disease detection application
Project by Fernando Herrán Albelda (Master in Data Science for KSchool). 

### Objective
This repository contains all the code and dependencies needed to build an Streamlit application. This application is part of the thesis done in the Master of Data Science for KSchool, which goal is to build a 3D - Convolutional Neural Network to classify patients with Alzheimer's disease (AD) or cognitively normal (CN) using Magnetic Resonance Images (MRI) of their brains. The Github repository of the project can be found [here](https://github.com/fernandoherran/master-thesis).

The streamlit application, which has been deployed using Azure, can be found [here](https://alzheimer-detection.herokuapp.com/).

### Set-up virtual environment
In order to run the code without having dependencies problems, the user can create a virtual environment with all the needed packages. To create the virtual environment, please follow process below: 

- Clone Github repository in your computer. Open terminal, and run comand below:
```
git clone https://github.com/fernandoherran/master-thesis.git
```
- Once cloned, run below command to create the virtual environment in your computer:
```
python -m venv virtual_env_streamlit
```
- Once created, the user can activate the virtual environment running the following command (first, the user must deactivate the current environment):
```
 source virtual_env_streamlit/bin/activate
```
- Finally, the user can install all the needed dependencies in the virtual environment running the following command (file requirements.txt has been downloaded when cloning the repository):
```
pip install -r requirements.txt
```

### Repository

File `tfm_app.py` contains the code to build the Streamlit application. It has some dependencies, which can be found in the folder *aux_dependencies*. In the main folder of the repository there are also the files Procfile and setup.sh, which are needed to deploy the application using Heroku, and the requirements.txt and README.md .
