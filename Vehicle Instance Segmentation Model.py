import cv2
import glob
import json
import numpy as np
import rasterio
from group_norm import GroupNormalization
from keras.models import Input, Model
from keras.layers import Activation, Add, AveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Nadam



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        image = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
    
    ulx = metadata['transform'][2]
    xres = metadata['transform'][0]
    uly = metadata['transform'][5]
    yres = metadata['transform'][4]
                                      
    lrx = ulx + (image.shape[1] * xres)                                                         
    lry = uly - (image.shape[0] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.shape[1]) ** 2 / (image.shape[1] + 1)) / (lrx - ulx)
        yf = ((image.shape[0]) ** 2 / (image.shape[0] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillPoly(mask, [position], 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, edge_array, image_height_size, image_width_size, mode, 
                                      percentage_overlap, buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image, its mask, and the mask edges in the form of fixed size 
    segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - edge_array: Numpy array representing the binary edge mask to mark out background and edge pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    - edge_segment_array: 4 - Dimensional numpy array containing the edge patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        edge_complete = np.zeros((y_size, edge_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
        edge_complete[0 : edge_array.shape[0], 0 : edge_array.shape[1], 0] = edge_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((mask_array.shape[0], x_size, 1))
        edge_complete = np.zeros((edge_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
        edge_complete[0 : edge_array.shape[0], 0 : edge_array.shape[1], 0] = edge_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        edge_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
        edge_complete[0 : edge_array.shape[0], 0 : edge_array.shape[1], 0] = edge_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = np.expand_dims(mask_array, axis = 2)
        edge_complete = np.expand_dims(edge_array, axis = 2)
        
    img_list = []
    mask_list = []
    edge_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
            edge_original = edge_complete[i : i + image_height_size, j : j + image_width_size, 0]
            edge_rotate_90 = cv2.warpAffine(edge_original, M_90, (image_height_size, image_width_size))
            edge_rotate_180 = cv2.warpAffine(edge_original, M_180, (image_width_size, image_height_size))
            edge_rotate_270 = cv2.warpAffine(edge_original, M_270, (image_height_size, image_width_size))
            edge_flip_hor = cv2.flip(edge_original, 0)
            edge_flip_vert = cv2.flip(edge_original, 1)
            edge_flip_both = cv2.flip(edge_original, -1)
            edge_list.extend([edge_original, edge_rotate_90, edge_rotate_180, edge_rotate_270, edge_flip_hor, edge_flip_vert, 
                              edge_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    edge_segment_array = np.zeros((len(edge_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        edge_segment_array[index, :, :, 0] = edge_list[index]
        
    return image_segment_array, mask_segment_array, edge_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    - edge_full_array: 4 - Dimensional numpy array containing binary edge mask patches extracted from all polygons for model 
                       training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + '\\Images' + '\\Image_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Polygons' + '\\Polygon_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    edge_array_list = []
    
    for file in range(len(img_files)):
        with rasterio.open(img_files[file]) as f:
            metadata = f.profile
            img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
            
        mask = training_mask_generation(img_files[file], polygon_files[file])
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros((mask.shape), dtype = np.uint8)
        for contour in contours:
            for edge_array in contour:
                edge[edge_array[0][1], edge_array[0][0]] = 1
                
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array, edge_array = image_clip_to_segment_and_convert(img, mask, edge, img_height_size, 
                                                                                  img_width_size, mode = 0, 
                                                                                  percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array, edge_array = image_clip_to_segment_and_convert(img, mask, edge, img_height_size, 
                                                                                  img_width_size, mode = 1, 
                                                                                  percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array, edge_array = image_clip_to_segment_and_convert(img, mask, edge, img_height_size, 
                                                                                  img_width_size, mode = 2, 
                                                                                  percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array, edge_array = image_clip_to_segment_and_convert(img, mask, edge, img_height_size, 
                                                                                  img_width_size, mode = 3, 
                                                                                  percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        edge_array_list.append(edge_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    edge_full_array = np.concatenate(edge_array_list, axis = 0)
    
    return img_full_array, mask_full_array, edge_full_array



def residual_block(input_tensor, num_of_filters, kernel_size, group_filters, residual_name):
    """
    This function is used to create a residual block as proposed in the paper 'Vehicle Instance Segmentation From Aerial Image
    and Video Using a Multitask Learning Residual Fully Convolutional Network' by Mou L., Zhu X.X. (2018).
    
    Inputs:
    - input_tensor: Input to the residual block
    - num_of_filters: Number of feature maps to be generated for the residual block
    - kernel_size: Size of convolutional kernel to be used for the residual block
    - group_filters: Number of groups to be used for group normalization
    - residual_name: Prefix to be appended to the name of each convolutional layer in the residual block
    
    Outputs:
    - residual_output: Output of the residual block
    
    """
    
    conv_1 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = residual_name + '_conv_1')(input_tensor)
    conv_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_1)
    conv_1_act = Activation('relu')(conv_1_gn)
    
    conv_2 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = residual_name + '_conv_2')(conv_1_act)
    conv_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_2)
    conv_2_act = Activation('relu')(conv_2_gn)
    
    conv_3 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = residual_name + '_conv_3')(conv_2_act)
    conv_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_3)
    
    
    
    conv_res = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                      name = residual_name + '_conv_residual')(input_tensor)
    conv_res_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_res)
    
    
    
    residual_output = Add()([conv_3_gn, conv_res_gn])
    
    
    
    return residual_output



def identity_block(input_tensor, num_of_filters, kernel_size, group_filters, identity_name):
    """
    This function is used to create an identity block as proposed in the paper 'Vehicle Instance Segmentation From Aerial Image
    and Video Using a Multitask Learning Residual Fully Convolutional Network' by Mou L., Zhu X.X. (2018).
    
    Inputs:
    - input_tensor: Input to the identity block
    - num_of_filters: Number of feature maps to be generated for the identity block
    - kernel_size: Size of convolutional kernel to be used for the identity block
    - group_filters: Number of groups to be used for group normalization
    - identity_name: Prefix to be appended to the name of each convolutional layer in the identity block
    
    Outputs:
    - identity_output: Output of the identity block
    
    """
    
    conv_1 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = identity_name + '_conv_1')(input_tensor)
    conv_1_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_1)
    conv_1_act = Activation('relu')(conv_1_gn)
    
    conv_2 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = identity_name + '_conv_2')(conv_1_act)
    conv_2_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_2)
    conv_2_act = Activation('relu')(conv_2_gn)
    
    conv_3 = Conv2D(num_of_filters, (kernel_size, kernel_size), padding = 'same', 
                    name = identity_name + '_conv_3')(conv_2_act)
    conv_3_gn = GroupNormalization(groups = group_filters, axis = -1, epsilon = 0.1)(conv_3)
    
    
    
    identity_output = Add()([input_tensor, conv_3_gn])
    
    
    
    return identity_output



def MLRFCN_Model(img_height_size, img_width_size, n_bands, gf, initial_conv_filter_numbers, kernel = 3, stage_1_filters = 256, 
                 stage_2_filters = 512, stage_3_filters = 1024, stage_4_filters = 2048, l_r = 0.0001, lamb = 0.1):
    """
    This function generates the Multitask Learning Residual Fully Convolutional Network (MLRFCN) as proposed in the paper 
    'Vehicle Instance Segmentation From Aerial Image and Video Using a Multitask Learning Residual Fully Convolutional 
    Network' by Mou L., Zhu X.X. (2018).
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - gf: Number of groups to be used for the group normalization
    - initial_conv_filter_numbers: Number of feature maps to be used for initial convolution
    - kernel: Kernel size to be used throughout each stage
    - stage_1_filters: Number of feature maps to be used throughout stage 1
    - stage_2_filters: Number of feature maps to be used throughout stage 2
    - stage_3_filters: Number of feature maps to be used throughout stage 3
    - stage_4_filters: Number of feature maps to be used throughout stage 4
    - l_r: Learning rate to be used for the Nesterov Adam optimizer
    - lamb: Lambda value to be used as weight for the edge binary crossentropy loss
    
    Outputs:
    - mlrfcn_model: MLRFCN model to be trained using input parameters and network architecture
    
    """
    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    
    
    
    conv_1 = Conv2D(initial_conv_filter_numbers, (7, 7), strides = (2, 2), padding = 'same', name = 'conv_1')(img_input)
    maxpool_1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(conv_1)
    
    
    
    res_block_1 = residual_block(maxpool_1, stage_1_filters, kernel, gf, 'res_block_1')
    identity_block_1_1 = identity_block(res_block_1, stage_1_filters, kernel, gf, 'ident_block_1_1')
    identity_block_1_2 = identity_block(identity_block_1_1, stage_1_filters, kernel, gf, 'ident_block_1_2')
    
    
    
    res_block_2 = residual_block(identity_block_1_2, stage_2_filters, kernel, gf, 'res_block_2')
    identity_block_2_1 = identity_block(res_block_2, stage_2_filters, kernel, gf, 'ident_block_2_1')
    identity_block_2_2 = identity_block(identity_block_2_1, stage_2_filters, kernel, gf, 'ident_block_2_2')
    identity_block_2_3 = identity_block(identity_block_2_2, stage_2_filters, kernel, gf, 'ident_block_2_3')
    identity_block_2_4 = identity_block(identity_block_2_3, stage_2_filters, kernel, gf, 'ident_block_2_4')
    identity_block_2_5 = identity_block(identity_block_2_4, stage_2_filters, kernel, gf, 'ident_block_2_5')
    avgpool_1 = AveragePooling2D(pool_size = (2, 2))(identity_block_2_5)
    
    
    
    res_block_3 = residual_block(avgpool_1, stage_3_filters, kernel, gf, 'res_block_3')
    identity_block_3_1 = identity_block(res_block_3, stage_3_filters, kernel, gf, 'ident_block_3_1')
    identity_block_3_2 = identity_block(identity_block_3_1, stage_3_filters, kernel, gf, 'ident_block_3_2')
    identity_block_3_3 = identity_block(identity_block_3_2, stage_3_filters, kernel, gf, 'ident_block_3_3')
    identity_block_3_4 = identity_block(identity_block_3_3, stage_3_filters, kernel, gf, 'ident_block_3_4')
    identity_block_3_5 = identity_block(identity_block_3_4, stage_3_filters, kernel, gf, 'ident_block_3_5')
    identity_block_3_6 = identity_block(identity_block_3_5, stage_3_filters, kernel, gf, 'ident_block_3_6')
    identity_block_3_7 = identity_block(identity_block_3_6, stage_3_filters, kernel, gf, 'ident_block_3_7')
    identity_block_3_8 = identity_block(identity_block_3_7, stage_3_filters, kernel, gf, 'ident_block_3_8')
    avgpool_2 = AveragePooling2D(pool_size = (2, 2))(identity_block_3_8)
    
    
    
    res_block_4 = residual_block(avgpool_2, stage_4_filters, kernel, gf, 'res_block_4')
    identity_block_4_1 = identity_block(res_block_4, stage_4_filters, kernel, gf, 'ident_block_4_1')
    identity_block_4_2 = identity_block(identity_block_4_1, stage_4_filters, kernel, gf, 'ident_block_4_2')
    avgpool_3 = AveragePooling2D(pool_size = (2, 2))(identity_block_4_2)
    
    
    
    stage_2_conv = Conv2D(1, (1, 1), padding = 'same', name = 'stage_2_conv')(avgpool_1)
    stage_2_upsam = UpSampling2D(size = (8, 8), interpolation = 'bilinear')(stage_2_conv)
    
    stage_3_conv = Conv2D(1, (1, 1), padding = 'same', name = 'stage_3_conv')(avgpool_2)
    stage_3_upsam = UpSampling2D(size = (16, 16), interpolation = 'bilinear')(stage_3_conv)
    
    stage_4_conv = Conv2D(1, (1, 1), padding = 'same', name = 'stage_4_conv')(avgpool_3)
    stage_4_upsam = UpSampling2D(size = (32, 32), interpolation = 'bilinear')(stage_4_conv)
    
    
    mask_output_inter = Add()([stage_2_upsam, stage_3_upsam, stage_4_upsam])
    mask_output = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid', name = 'mask')(mask_output_inter)
    
    edge_output_inter = Add()([stage_2_upsam, stage_3_upsam, stage_4_upsam])
    edge_output = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid', name = 'edge')(edge_output_inter)
    
    
    mlrfcn_model = Model(inputs = img_input, outputs = [mask_output, edge_output])
    mlrfcn_model.compile(loss = ['binary_crossentropy', 'binary_crossentropy'], loss_weights = [1.0, lamb],
                         optimizer = Nadam(lr = l_r), metrics = ['binary_crossentropy'])
    
    
    return mlrfcn_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
     
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds, _ = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0] = preds[0, :, :, 0]
            
    mask_complete = np.expand_dims(mask[0 : img.shape[0], 0 : img.shape[1], 0], axis = 2)
    mask_complete = np.transpose(mask_complete, [2, 0, 1]).astype('float32')
    
    
    if write:
        metadata['count'] = 1
        metadata['dtype'] = 'float32'
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete
