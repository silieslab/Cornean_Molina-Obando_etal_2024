
# Connectome Analysis

This README file contains essential information for reproducing the analysis and plots related to EM figures in Cornean, Molina-Obando et al 2024.

## Code for the Analysis and Data Visualization:

### 1. Variability-analysis-main:

This script contains the main analysis for RAW_DATA. It saves PROCESSED_DATA files for subsequent steps. During the analysis, statistical summaries of interest, as mentioned in the main publication paper, are printed. Postsynaptic-specific paper figures are also generated. It takes input from Excel and TXT files for a particular postsynaptic neuron (e.g., Tm9_neurons_input_count_L_2023082, Tm9_proofreadings_20230823, and Tm9_L.txt).

### 2. Variability-postanalysis-main:

This script contains the main analysis for PROCESSED_DATA. Comparative analysis and figures across datasets (across optic lobes or across different postsynaptic neurons) are generated here. It takes input from various Excel files for different postsynaptic neurons (e.g., Tm9_FAFB_R.xlsx, Tm1_FAFB_R.xlsx, and Tm2_FAFB_R.xlsx).

### 3. Helper:

This script contains custom functions used in the main and postanalysis pipelines.

### 4. Twig-proofreading-analysis-main:

This script contains the analysis of RAW_DATA and plots twig proofreading supplementary figures. It takes input from an Excel file for a particular postsynaptic neuron (e.g., All_Tm9_neurons_input_count_ME_L_impact_twig_proofreading_20230718.xlsx).

## Data Structure for Analysis and Plot:

For a smooth execution, it is recomended to respect the following data structure of the main folder:

1. Raw-data (all raw data files inside)
    - 1.1 All_Tm9_neurons_input_count_ME_L_impact_twig_proofreading_20230718.xlsx
    - 1.2 Tm9_proofreadings_20230823
    - ...
2. Pdf-plots (plots saved here)
3. Processed-data (all processed files inside)
    - 3.1 Tm9_FAFB_R.xlsx
    - 3.2 Tm9_FAFB_L.xlsx
    - ...
    - 3.6 Figures (plots saved here)
4. Txts (all optic lobe ID files inside)
    - 4.1 Tm9_L.txt
    - 4.2 Tm9_L_R.txt
    - ...
    - 4.5 Tm2_R.txt

## Software Requisites:

### Environment Manager:

- Download and install Anaconda (or Miniconda) from [Anaconda Installation](https://docs.anaconda.com/anaconda/install/).

### Initialize Environment:

The Anaconda prompt or Git Bash terminals can be used. In Git Bash, Anaconda environments need to be set as the source by running:

```bash
source C:/Users/[USERNAME]/anaconda3/Scripts/activate

conda create --name <env_name> python=3.9
activate <env_name>


pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
# (other packages might be needed)

```

Recommended user interfaces to run the scripts:
- Atom: https://atom.io/
- VSCode: https://code.visualstudio.com/

