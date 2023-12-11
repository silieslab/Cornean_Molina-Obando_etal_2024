# -*- coding: utf-8 -*-
"""
Created on Fr Sep 30 11:30 2022

@author: Jacqueline Cornean


Here are the functions used in the expansion microscopy analysis.

"""

from tqdm.notebook import tqdm
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from skimage.feature import match_template, peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
import matplotlib.colors as mcolors
import random
import matplotlib.pyplot as plt


#%%
def thresholding_and_props(channel_data, threshold = None):
    '''
    This function reduces noise by using an otsu threshold to get the right thresholding value.
    Then, we will use this threshold value to threshold our data and get rid of some noise.
    This, we can do for each channel on its own.
    
    Parameters:
    ================
    channel_data: array
                input image/ array from one channel (e.g image_part_brp or image_part_gfp)
    
    threshold: int or bool
                if None, the intensity threshold will be calculated via otsu, else it will
                take the input integer as threshold
    
    Returns:
    ===============
    region_props_channel: dict
                a dict with the properties of the before thresholded and labeled regions
                Properties: 'label': indices of the regions (should be the same as in the input of this function)
                            'area': area of each region
                            'coords': coordinates of each region
    '''
    if threshold == None:
        threshold = threshold_otsu(channel_data)
    binary_array = channel_data > threshold #array of booleans for the ones above threshold
    region_labels = label(binary_array) #labels connected regions of an integer array
    region_props_channel = regionprops_table(region_labels, properties = ('label', 'area','coords'))
    ##regionprops_table() can use the labeled input image and get different properties out of them
    
    return region_props_channel

def make_mask (channel_data, region_props_channel, region_count = -50):
    '''
    Function to generate a binary mask by using the thresholded values. First, there is an array with zeros, which will be filled with
    ones for regions that are above the previous otsu thresholding (see the function ``thresholding_and_props``).

    Parameters:
    ===========
    channel_data: array
        input image/ array from one channel (e.g image_part_brp or image_part_gfp)

    region_props_channel: dict
        a dict with the properties of the before thresholded and labeled regions
        Properties: 'label': indices of the regions (should be the same as in the input of this function)
                            'area': area of each region
                            'coords': coordinates of each region

    region_count: int
        Can be used to threshold over the area size of the regions. If one wants only the 50 biggest regions, region_count would be -50.
        If one wants all regions, region_count would be 1.

    Returns:
    =========
    new_mask: array
        binary mask of the image

    '''
    new_mask = np.zeros(channel_data.shape, dtype='?')
    #fill in using region coordinates
    area_threshold = np.sort(region_props_channel['area'])[region_count] - 1
    biggest_area = np.where(region_props_channel['area'] > area_threshold)
    #just take the 50 biggest regions
    for x in biggest_area[0]:
        new_mask[region_props_channel['coords'][x][:,0],region_props_channel['coords'][x][:,1],
                    region_props_channel['coords'][x][:,2]] = int(1)
    return new_mask

def calculate_local_max (image, min_distance, thresholds, filename, size_z):
    '''
    Calculate the local maxima of the Brp image to get all Brp puncta.

    Parameters:
    ===========
    image: array
        ROI or thresholded image that has the original values in the non background regions (e.g. brp_masked_stack).
        We will calculate the local maxima of Brp within this region.

    min_distance: int
        the minimal allowed distance separating the peaks of lokal maxima

    thresholds: int
        minimum luminance intensity of peaks

    filename: str
        can be 'brp' or 'gfp' but most of the times 'brp'
        This can indicate later which channel one used to extract the local maxima.
        
    size_z: int
        factor of the ratio between x and/or y and z orientation
        (in image properties --> pixel Size/ Voxel Size values for x and z: z value/ x value = size_z)

    Returns:
    ==========
    results: list of dict
        coordinates of every local maxima
    '''
    results = []
    local_max_indices = peak_local_max(image, min_distance, thresholds)
    locmax_pd = pd.DataFrame(local_max_indices)
    locmax_pd = locmax_pd*[size_z,1,1]+[2,0,0]
    locmax_pd['number_of_dots'] = local_max_indices.shape[0]

    locmax_pd['file'] = filename

    results.append(locmax_pd)
    return (results)

def points_for_napari (puncta_coords):
    '''
    Parameters:
    ===========
    puncta_coords: np array
                coordinates of each puncta (local maxima of intensity) for the input channel.
                
    Returns:
    ===========
    columns: list
                make a list, each element of the list is one column of the puncta_coords.
                That means that each element of the list is one axis of space (z, x, y) and the same index of each element
                concludes the coordinates of one point.
                This is the structure that Napari needs to load the points.
    '''
    columns = []
    for coord in range(3):
        curr_column = puncta_coords[:,coord]
        columns.append(curr_column)
    return columns

#%%
def get_gfp_label_coord (new_mask_gfp, size_z):
    '''
    Get coordinates of every pixel that has a GFP signal.
    Parameters:
    ===========
    new_mask_gfp: array
        binary GFP mask which has boolean entries

    size_z: int
        factor of the ratio between x and/or y and z orientation
        (in image properties --> pixel Size/ Voxel Size values for x and z: z value/ x value = size_z)
    
    Returns:
    ===========
    label_coord_dict: dict
        all coodrinates of the GFP signal
    '''
    label_coord_dict = {}
    binary_gfp_mask = new_mask_gfp * 1
    curr_label = np.where(binary_gfp_mask == 1)
    curr_label_df = pd.DataFrame(list(zip(*curr_label)), columns = ['z', 'x', 'y']) #need to transpose tuple to make array with the 3 coords
    curr_label_df = curr_label_df*[size_z,1,1]+[2,0,0] #correct with z_size
    label_coord_dict[1] = curr_label_df

    return label_coord_dict

#%%
def get_all_brps_in_ROI (brp_masked_stack, roi_coords, image_part_brp, min_distance, thresholds, size_z):
    '''
    Get all Brp puncta by calculating the local maxima of the Brp signal for each manually drawn Brp ROI.
    By extracting the local maxima of each ROI, we get the puncta located in single columns.

    Parameters:
    ============
    brp_masked_stacked: array
        masked image of the Brp channel. Everything outside of the masks is zero (background) and inside are the true values of the raw image.

    roi_coord: dict
        coordinates of each manually drawn ROI

    image_part_brp: array
        sliced raw image that was used until now

    min_distance: int
        the minimal allowed distance separating the peaks of lokal maxima

    thresholds: int
        minimum intensity of peaks
        
    size_z: int
        factor of the ratio between x and/or y and z orientation
        (in image properties --> pixel Size/ Voxel Size values for x and z: z value/ x value = size_z)

    Returns:
    ===========
    rois_df: DataFrame
        For each ROI we have the coordinates and respective napari points (coordinates in a format that napari 
        can display the points) of all found local maxima in each ROI.

    '''
    roi_list = []
    for rois in range (len(roi_coords)):
        x_start = int(roi_coords['X'][rois] - (roi_coords['W'][rois]/2))
        x_end = int(roi_coords['X'][rois] + (roi_coords['W'][rois]/2))
        y_start = int(roi_coords['Y'][rois] - (roi_coords['H'][rois]/2))
        y_end = int(roi_coords['Y'][rois] + (roi_coords['H'][rois]/2))

        zeros = np.zeros(image_part_brp.shape)
        zeros[:,y_start:y_end,x_start:x_end]=1
        roi0 = zeros*brp_masked_stack
        brp_max = calculate_local_max(roi0, min_distance, thresholds, 'brp', size_z) #brp_mask_part, brp_masked_stack
        all_results = pd.concat(brp_max)
        brp_coords = np.array(all_results[[0,1,2]])
        brp_puncta = points_for_napari(brp_coords)

        roi_list.append({'cluster': rois, 'coords': brp_coords, 'points': brp_puncta})
    rois_df = pd.DataFrame(roi_list, columns = ['cluster', 'coords', 'points'])

    return rois_df

#%%
def get_closest_brp_from_brp_clusters (new_mask_gfp, size_z, rois_df, distance_brp_gfp):
    '''
    Calculate closest brp puncta to each Tm9-GFP signal in Brp based ROIs.
    Parameters:
    ===========
    new_mask_gfp: array
        binary GFP mask which has boolean entries

    size_z: int
        factor of the ratio between x and/or y and z orientation
        (in image properties --> pixel Size/ Voxel Size values for x and z: z value/ x value = size_z)

    rois_df: DataFrame
        For each ROI we have the coordinates and respective napari points (coordinates in a format that napari 
        can display the points) of all detected local maxima in each ROI.
    
    distance_brp_gfp: int
        permissive distance between gfp and brp signal
        int(0.155 * exp_factor / pixel_size_x)

    Returns:
    ==========
    distance_df: DataFrame
        For each cluster/ ROI we get the indices of the coordinates of the GFP-Brp pair that fulfilled the distance threshold.
        In addition, the distance between the two coordinates is listed in pixel.
    '''
    #make a df with the closest Brp puncta indicating each cluster
    closest_labels_list = []
    label_coord_dict = get_gfp_label_coord (new_mask_gfp, size_z)
    for cluster in range(len(rois_df)):
        for brp_index, point in enumerate(rois_df['coords'][cluster]):
            dist = cdist([point], label_coord_dict[1])
            gfp_index = np.where(dist[0] == np.min(dist[0]))[0][0]
            distance = dist[0][gfp_index]
            if distance <= distance_brp_gfp:
                closest_labels_list.append({'cluster': cluster, 'gfp_index': gfp_index, 'brp_index': brp_index, 'distance': distance})
    distance_df = pd.DataFrame(closest_labels_list, columns = ['cluster', 'gfp_index', 'brp_index', 'distance'])

    return distance_df

#%%
def get_closest_brp_per_roi (new_mask_gfp, roi_coords, image_part_gfp, brp_coords, distance_brp_gfp, size_z):
    
    '''
    Calculate closest Brp puncta to each Tm9-GFP ROI.
    
    Parameters:
    ===========
    new_mask_gfp: array
            binary GFP mask which has boolean entries

    roi_coords: dict
            coordinates of each manually drawn ROI
    
    image_part_gfp: array
            part of the raw image that corresponds to new_gfp_mask

    brp_coords: array
            coordinates of all brp points found by the local maximum detection.

    size_z: int
            factor of the ratio between x and/or y and z orientation
            (in image properties --> pixel Size/ Voxel Size values for x and z: z value/ x value = size_z)
    
    distance_brp_gfp: int
            permissive distance between gfp and brp signal.
            int(0.155 * exp_factor / pixel_size_x)

    Returns:
    ===========
    distance_df: DataFrame
            For each cluster/ ROI we get the indices of the coordinates of the GFP-Brp pair that fulfilled the distance threshold.
            In addition, the distance between the two coordinates is listed in pixel.
    '''
    
    binary_gfp_mask = new_mask_gfp * 1
    label_coord_dict = {}
    for rois in range (len(roi_coords)):
        x_start = int(roi_coords['X'][rois] - (roi_coords['W'][rois]/2))
        x_end = int(roi_coords['X'][rois] + (roi_coords['W'][rois]/2))
        y_start = int(roi_coords['Y'][rois] - (roi_coords['H'][rois]/2))
        y_end = int(roi_coords['Y'][rois] + (roi_coords['H'][rois]/2))

        zeros = np.zeros(image_part_gfp.shape)
        zeros[:,y_start:y_end,x_start:x_end] = 1
        roi0 = zeros * binary_gfp_mask

        curr_label = np.where(roi0 == 1)
        curr_label_df = pd.DataFrame(list(zip(*curr_label)), columns = ['z', 'x', 'y']) #need to transpose tuple to make array with the 3 coords
        curr_label_df = curr_label_df*[size_z,1,1]+[2,0,0] #correct with z_size
        label_coord_dict[rois] = curr_label_df

    closest_labels_list = []
    for rois in range (len(roi_coords)):
        if len(label_coord_dict[rois]) == 0:
            continue
        for brp_index, point in enumerate(brp_coords):
            dist = cdist([point], label_coord_dict[rois])
            gfp_index = np.where(dist[0] == np.min(dist[0]))[0][0]
            distance = dist[0][gfp_index]
            if distance <= distance_brp_gfp:
                closest_labels_list.append({'cluster': rois, 'gfp_index': gfp_index, 'brp_index': brp_index, 'distance': distance})
    distance_df = pd.DataFrame(closest_labels_list, columns = ['cluster', 'gfp_index', 'brp_index', 'distance'])

    return distance_df


#%%
def get_synapse_counts (roi_coords, distance_df, viewer, rois_df = None, brp_coords = None, roi_type = 'brp', point_size = 11):
    '''
    Parameters:
    ============
    roi_coord: dict
        coordinates of each manually drawn ROI

    distance_df: DataFrame
        For each cluster/ ROI we get the indices of the coordinates of the GFP-Brp pair that fulfilled the distance threshold.
        In addition, the distance between the two coordinates is listed in pixel.

    viewer: 
        napari viewer

    rois_df: DataFrame
        For each ROI we have the coordinates and respective napari points (coordinates in a format that napari 
        can display the points) of all found local maxima in each ROI.

    brp_coords: array
        coordinates of all brp points found by the local maximum detection.

    roi_type: str
        'brp' if you used the Brp channel to select ROIs
        'gfp' if you used the GFP channel to select ROIs

    point_size: int
        size of the synapse points displayed in the napari viewer

    Returns:
    ===========
    nearest_synapse_count: list
        list of total synapses per column 
    '''
    #make a random color palette
    colour_names = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colour_names)
    nearest_synapse_count = []
    for cluster in range(len(roi_coords)):
        curr_cluster = distance_df.loc[distance_df['cluster'] == cluster] #get all closest Brp indices per cluster
        count = len(curr_cluster) #get number of puncta per cluster
        nearest_synapse_count.append(count) #make list of these numbers
        if count == 0:
            continue
        cluster_points = [] #list to store coordinates of closest points in
        for index_closest_brp in curr_cluster['brp_index']: #iterate over indices of closest brp
            if roi_type == 'brp':
                curr_coords = rois_df['coords'][cluster][index_closest_brp]
            else:
                curr_coords =  brp_coords[index_closest_brp]
            cluster_points.append(curr_coords)
        points = points_for_napari(np.array(cluster_points)) #change coords back to point for napari
        colour = colour_names[cluster] #get a new color for each cluster
        viewer.add_points((np.array(points)).T, scale=(1, 1, 1), size = point_size, face_color = colour)

    return nearest_synapse_count

#%%
def plot_colourcoded_puncta_napari (rois_df, viewer):
    '''
    Plot all Brp puncta for each ROI in napari.
    Parameters:
    ===========
    rois_df: DataFrame
        For each ROI we have the coordinates and respective napari points (coordinates in a format that napari 
        can display the points) of all detected local maxima in each ROI.

    Returns:
    ===========
    Output in napari viewer.
    '''
    #make a random color palette
    colour_names = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colour_names)
    colour_names[1]

    for index in range(len(rois_df)):
        colour = colour_names[index]
        viewer.add_points((np.array(rois_df['points'][index])).T, scale = (1, 1, 1), size = 14, face_color = colour)