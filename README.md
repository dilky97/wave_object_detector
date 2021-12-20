# Object Detector

Object Detection app using Detectron2 and [H2O Wave](https://wave.h2o.ai)


<img src="./static/demo.gif" width="100%" height="100%"/>

## Running this App Locally

### System Requirements 
1. Python 3.6+
2. pip3

### 1. Download the Wave Server
Follow the documentation to [download and run](https://wave.h2o.ai/docs/installation) the Wave Server on your local machine.<br>


### 2. Build the python environment
Clone the repo and create python environment with the required dependencies.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
You might need to follow the official instructions to install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

### 3. Run the application
```
wave run src.app
```
### 4. View the application
Go to http://localhost:10101 from browser
