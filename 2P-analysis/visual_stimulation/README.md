# Visual stimulation

Data and code to generate the Figure on response variability in Tm9 neurons to ONOFF fullfield flashes. Responses were recorded with in vivo two-photon calcium imaging using GCaMP6f.
Responses are quantified as dF/F0, over manually selected regions of interest (ROIs).
To reduce the effect of response amplitude on the characterization of the shape of the response time-courses, the traces are normalized per ROI as the z-score (x -  mean(x)) / std(x).

The z-scored responses were clustered by k-means with a cluster number of 6, using the correlation distance, in MATLAB.

Clusters are visualized by the z-scored traces belonging to each cluster.

Dimensionality reduction is applied for the z-scored traces, PCA and t-SNE are shown with points colored by cluster membership.

## Code

To generate figures used in the paper, first download the full folder, and start matlab from inside the directory so all paths are added properly. Final figures require additional vector illustration edits.

## Data

Data contained in the data folder are MAT-files with the following structure

Variables are contained in a table with dimensions (number of recorded layers, number of data variables)

An example table entry (row) will consist of 12 data variables (columns):

- flyInd: Index of the fly

- stackInd: Index of the layer recorded within a fly

- timeSeriesPath: Original path to motion corrected data, aligned to stimulus timing in local drive

- stimParams: Set of parameters defining the stimulus used, one column per stimulus epoch, e.g., one bar position is a different epoch with all parameters unchanged except its position. Parameter examples: stimtype, lum, contrast, duration, stimrot, spacing, randomize, USEFRUSTUM (perspective correction).

- stimParamFileName: Filename of used stimulus file

- typeOfProblem: Zero for no problem, otherwise indicates data that could not be processed by the analysis code, potentially incomplete or aborted recording.

- cycleInd: Each stack corresponds to a layer, while each cycle corresponds to a time series within that layer. Old Tm9 data in figure 2b, was acquired using multiple stacks to record the same layer each stack has two cycles, an empty, prestimulus cycle and a stimulus cycle. The remaining datasets were acquired using a single stack per layer, and one cycle per stimulus, with no prestimulus cycles.

- nonParamTuning: array structure (dimensions: 1, number of channels) with tuning curves obtained via different methods (mean, variance, min, max, etc), resulting tuning curves are stored in tc variable with dimensions (number of tuning methods), each entry contains the tuning curves for that method with dimensions (1, number of ROIs, number of stimulus epochs excluding the interstimulus interval epoch). Domain indicated the stimulus parameter dimension of the tuning, in the case of bars it is the bar position measured in degrees.
  
  ```
  ANOVApMap_Responsive: {1×10 cell}
     ANOVApMap_Selective: {1×10 cell}
               epochMaps: {{29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}  {29×1 cell}}
                      tc: {[1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]  [1×9×29 double]}
                tcParams: {[1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]  [1×1 struct]}
                  domain: [-28 -26 -24 -22 -20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28]
          originalDomain: [-28 -26 -24 -22 -20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28]
              stimSizeTC: [29 1]
            fieldNamesTC: {'stimtrans_amp'}
        stimParamsFlatTC: [1×29 struct]
          collapseMethod: 'nanmean'
           tuningMethods: {'mean'  'var'  'min'  'max'  'pctile5'  'median'  'pctile95'  'extreme'  'SNR'  'gain'}
  ```

- paramTuning: struct with dimensions (channels, number of ROIs). 
  
  - Fields: 
    
    - fit: structure containing parameters from Gaussian fit to spatial receptive fields
      
      ```
      General model Gauss1:
      ans(x) =  a1*exp(-((x-b1)/c1)^2)
      Coefficients (with 95% confidence bounds):
       a1 =      0.2211  (0.175, 0.2673)
       b1 =        18.2  (17.84, 18.56)
       c1 =       2.117  (1.608, 2.627)
      ```
    
    - gof: struct with different measures of the goodness of fit
      
      ```
       sse: 0.4333
      rsquare: 0.5321
       dfe: 113
      adjrsquare: 0.5238
      rmse: 0.0619
      ```
    
    - info: struct with information about fitting procedure, not relevant.

- roiMeta: array structure (dimensions: 1, number of channels)  with fields backgroundRois (index of ROIs used for background subtraction, selected at last), invalidRois (indices of ROIs with NaNs in the data or some other issue).

- responseIndex: array with dimensions (channels, number of ROIs) containing the response quality index per ROI.

- trialsPValue: array with dimensions (channels, number of ROIs) containing the pValue testing for trial to trial differences per ROI. Not relevant for the paper analysis.

```
ProcessedTable =

  187×12 table

    flyInd    stackInd                                   timeSeriesPath                                    stimParams                                stimParamFileName                               typeOfProblem    cycleInd    nonParamTuning     paramTuning       roiMeta       responseIndex    trialsPValue 
    ______    ________    ____________________________________________________________________________    _____________    ______________________________________________________________________    _____________    ________    ______________    _____________    ____________    _____________    _____________

     1        1           'D:\Data_LGRTL\Two_photonMat\190428.0.lr\190428.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×4  struct]    'FullField_ONOFF_1.0_2s_BG_0.5_4s_Weber_NonRand.txt'                      0                1           [        -1]      [         -1]    [1×1 struct]    [1×8  double]    [1×8  double]
     1        1           'D:\Data_LGRTL\Two_photonMat\190428.0.lr\190428.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×30 struct]    'StandingStripe_1s_YAxis_5degWide_2degSep_m1.0Con_rand_USEFRUSTUM.txt'    0                2           [1×1 struct]      [1×8  struct]    [1×1 struct]    [1×8  double]    [1×8  double]
     1        1           'D:\Data_LGRTL\Two_photonMat\190428.0.lr\190428.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×30 struct]    'StandingStripe_1s_YAxis_5degWide_2degSep_p1.0Con_rand_USEFRUSTUM.txt'    0                3           [1×1 struct]      [1×8  struct]    [1×1 struct]    [1×8  double]    [1×8  double]
     1        1           'D:\Data_LGRTL\Two_photonMat\190428.0.lr\190428.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×26 struct]    'StandingStripe_1s_XAxis_5degWide_2degSep_m1.0Con_rand_USEFRUSTUM.txt'    0                4           [1×1 struct]      [1×8  struct]    [1×1 struct]    [1×8  double]    [1×8  double]
     1        1           'D:\Data_LGRTL\Two_photonMat\190428.0.lr\190428.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×26 struct]    'StandingStripe_1s_XAxis_5degWide_2degSep_p1.0Con_rand_USEFRUSTUM.txt'    0                5           [1×1 struct]      [1×8  struct]    [1×1 struct]    [1×8  double]    [1×8  double]
     2        1           'D:\Data_LGRTL\Two_photonMat\190429.0.lr\190429.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×4  struct]    'FullField_ONOFF_1.0_2s_BG_0.5_4s_Weber_NonRand.txt'                      0                1           [        -1]      [         -1]    [1×1 struct]    [1×8  double]    [1×8  double]
     2        1           'D:\Data_LGRTL\Two_photonMat\190429.0.lr\190429.0.lr_fly1_Tm1GCaMP6f\TS-001'    [1×26 struct]    'StandingStripe_1s_XAxis_5degWide_2degSep_m1.0Con_rand_USEFRUSTUM.txt'    0                2           [1×1 struct]      [1×8  struct]    [1×1 struct]    [1×8  double]    [1×8  double]
```

## Fly IDs and timeseries used here

'D:\Data_LGRTL\Two_photonMat\160428.0.lr\160428.0.lr_fly1_Tm9GCaMP6f\TSeries-04282016-1239-008'
'D:\Data_LGRTL\Two_photonMat\160428.0.lr\160428.0.lr_fly1_Tm9GCaMP6f\TSeries-04282016-1239-009'
'D:\Data_LGRTL\Two_photonMat\160428.0.lr\160428.0.lr_fly1_Tm9GCaMP6f\TSeries-04282016-1239-019'
'D:\Data_LGRTL\Two_photonMat\160429.0.lr\160429.0.lr_fly1_Tm9GCaMP6f\TSeries-04292016-1219-003'
'D:\Data_LGRTL\Two_photonMat\160727.0.lr\160727.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160729.0.lr\160729.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160729.0.lr\160729.0.lr_fly1_Tm9GCaMP6f\TS-009'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly2_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly2_Tm9GCaMP6f\TS-005'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly3_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly3_Tm9GCaMP6f\TS-008'
'D:\Data_LGRTL\Two_photonMat\160801.0.lr\160801.0.lr_fly3_Tm9GCaMP6f\TS-011'
'D:\Data_LGRTL\Two_photonMat\160804.0.lr\160804.0.lr_fly2_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160804.0.lr\160804.0.lr_fly2_Tm9GCaMP6f\TS-011'
'D:\Data_LGRTL\Two_photonMat\160804.0.lr\160804.0.lr_fly2_Tm9GCaMP6f\TS-013'
'D:\Data_LGRTL\Two_photonMat\160804.0.lr\160804.0.lr_fly2_Tm9GCaMP6f\TS-016'
'D:\Data_LGRTL\Two_photonMat\160824.0.lr\160824.0.lr_fly1_Tm9GCaMP6f\TS-005'
'D:\Data_LGRTL\Two_photonMat\160824.0.lr\160824.0.lr_fly1_Tm9GCaMP6f\TS-010'
'D:\Data_LGRTL\Two_photonMat\160824.0.lr\160824.0.lr_fly2_Tm9GCaMP6f\TS-005'
'D:\Data_LGRTL\Two_photonMat\160824.0.lr\160824.0.lr_fly2_Tm9GCaMP6f\TS-012'
'D:\Data_LGRTL\Two_photonMat\160824.0.lr\160824.0.lr_fly3_Tm9GCaMP6f\TS-005'
'D:\Data_LGRTL\Two_photonMat\160828.0.lr\160828.0.lr_fly2_Tm9GCaMP6f\TS-005'
'D:\Data_LGRTL\Two_photonMat\160828.0.lr\160828.0.lr_fly2_Tm9GCaMP6f\TS-009'
'D:\Data_LGRTL\Two_photonMat\160830.0.lr\160830.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly1_Tm9GCaMP6f\TS-006'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly2_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly2_Tm9GCaMP6f\TS-006'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly3_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160831.0.lr\160831.0.lr_fly3_Tm9GCaMP6f\TS-008'
'D:\Data_LGRTL\Two_photonMat\160913.0.lr\160913.0.lr_fly2_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160913.0.lr\160913.0.lr_fly3_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160914.0.lr\160914.0.lr_fly1_Tm9GCaMP6f\TS-001'
'D:\Data_LGRTL\Two_photonMat\160914.0.lr\160914.0.lr_fly1_Tm9GCaMP6f\TS-007'
'D:\Data_LGRTL\Two_photonMat\160915.0.lr\160915.0.lr_fly1_Tm9GCaMP6f\TS-004'