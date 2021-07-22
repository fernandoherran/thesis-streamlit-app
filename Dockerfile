FROM python:3.7-slim

# Create a new user to avoid running the app as root
RUN useradd --create-home appuser

WORKDIR ./home/appuser

COPY requirements.txt ./

# install dependencies before copying the file so they're cached for future builds
RUN pip install -r requirements.txt

# Copy the rest of the app files
COPY . /home/appuser

# Copy streamlit config files
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

# Expose ports
EXPOSE 80

ENTRYPOINT ["streamlit", "run"]
CMD ["tfm_app.py"]