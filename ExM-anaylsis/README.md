# ExM_analysis
With this analysis, one can quantify synapses in brains enlarged by expansion microscopy.
To use this analysis, the imaged brains should have a presynaptic neuron, expressing Brp[short]::mCherry, and a postsynaptic partner, labeled with a second fluorescence marker like rCD2::GFP.
Using the fluorescence signals of these markers, we can calculate the distance between them and extract synapses through a proximity threshold.

## Environment:
We used anaconda to create the environment:
  - Download and install anaconda (or miniconda) from: https://docs.anaconda.com/anaconda/install/
    
**Create an environment by using conda and pip:**
-	``conda create --name exm_env python=3.9``
-	``conda activate exm_env``

**Dependencies:**
-	``pip install pyclesperanto-prototype==0.24.1``
-	``python -m pip install "napari[pyqt5]"``
-	``pip install PyQt5==5.15.9``
-	``pip install napari-pyclesperanto-assistant==0.22.1``
-	``pip install pydantic==1.10.9``
-	``pip install typing-extensions==4.6.3``
-	``pip install napari-roi==0.1.8``
-	``pip install jupyterlab==4.0.9``
-	``pip install pynrrd==1.0.0``
-	``pip install seaborn==0.12.2``

## Preprocessing:
Preprocessing was done by utilizing the VVDViewer (https://github.com/JaneliaSciComp/VVDViewer).
Please download the VVDViewer from https://github.com/JaneliaSciComp/VVDViewer and follow the installation instructions on their website. 
<br>During the preprocessing step, we select the area of interest, e.g. the dendritic part of Tm9 neurons for both the GFP and the Brp channels.
Furthermore, during this step we reduce the noise level of the image. 

<br>**How to use it:**
-	Load the microscope image as a .tif file
-	Change the x, y, z pixel size if needed. (Please confirm by comparing the values with the meta data.)
-	Click on the **Analye** button.
-	Choose the **Paint Brush** option.
-	Click on the **Select** button and change the brush size and threshold value accordingly.
-	Use the brush tool to select the area of interest.
-	After the selections and thresholding is done, save the created mask:
	  - Right click on the dataset you were working on --> *Export mask* --> save as .nrrd file.

## Run the code:
For information on how to execute the script, please follow the instructions in the jupyter notebook ``Expansion_main.ipynb``.
<br>Custom written functions, that are utilized by the main script are in ``expansion_analysis_function.py``.
