######################################################################################################################################################################################
#                                                               Whole Slide Images (WSI) preprocessing tools                                                                         #
#                                                                     by: Juan Sebastian Malagón Torres                                                                              #
######################################################################################################################################################################################


##**req install**##:

# FUnction designed to isntall neccesary tensorflow version and tool
def req_install():
    bash_comand = ['pip install openslide-python',
                 'apt-get install openslide-tools',
                 'pip install -U scikit-image',
                 'pip install numpy']
    
    for comand in bash_comand:
        os.system(comand)
      
    import openslide
    from skimage import exposure, filters
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import sys

#packages:
import os

try:
    import openslide
    from skimage import exposure, filters, morphology
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import sys
except:
    print('Intalling packages!')
    req_install()



#################################################################################### svs_read #######################################################################################

#svs_read:
#Open .SVS slide images using openslide.
##Inputs: 
#   *file_path(string): file path.
##Outputs: 
#   *WSI(slide): object with slides images and metadata.

def svs_read(file_path):
    WSI = openslide.OpenSlide(file_path)
    return(WSI)
    
################################################################################### blue_ratio #######################################################################################

#blue_ratio:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *equalize(Bool): apply histogram equalization to blue ratio images.
#   *binarized(Bool): apply  to blue ratio images.
##Outputs: 
#   *Br(array): a 2D array with Blue-ratio values.

def blue_ratio(image, equalize = False, binarized = False):
    image = np.uint16(image)
    image = 255*(image - np.min(image))/(np.max(image)-np.min(image))
    R,G,B = image[:,:,0], image[:,:,1], image[:,:,2]
    
    Br = (B)/((1+R+G)*(1+R+G+B)) 
    Br[Br>1]=np.max(Br[Br<=1])
    
    if equalize == True:
        Br = exposure.equalize_hist(Br)
    if binarized == True:
        umbral = filters.threshold_triangle(Br)
        Br = (Br>=umbral)
    return(Br)

################################################################################### tissue_segmentation ###################################################################################

#tissue_segmentation:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *equalize(Bool): apply histogram equalization to blue ratio images.
#   *binarized(Bool): apply  to blue ratio images.
##Outputs: 
#   *Br(array): a 2D array with Blue-ratio values.

def tissue_segmentation(image, method = filters.threshold_mean, footprint_size = None):
  R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
  
  R = (R - np.min(R))/(np.max(R) - np.min(R))
  G = (G - np.min(G))/(np.max(G) - np.min(G))
  B = (B - np.min(B))/(np.max(B) - np.min(B))

  comparative_0 = np.abs(R-G)
  comparative_1 = np.abs(R-B)
  comparative_2 = np.abs(G-B)
    
  prob_mask = comparative_0*comparative_1*comparative_2
  mask = prob_mask > method(prob_mask)

  footprint = np.ones([3,3]) 
  closed_mask = morphology.closing(mask, footprint)

  return(closed_mask)
    
################################################################################### marker_detection ######################################################################################

#blue_ratio:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *color(string): only available red, green and blue.
##Outputs: 
#   *percentage(float): marker percentage on the image  
    
PENS_RGB = {
    "red": [
        (150, 80, 90),
        (110, 20, 30),
        (185, 65, 105),
        (195, 85, 125),
        (220, 115, 145),
        (125, 40, 70),
        (200, 120, 150),
        (100, 50, 65),
        (85, 25, 45),
    ],
    "green": [
        (150, 160, 140),
        (70, 110, 110),
        (45, 115, 100),
        (30, 75, 60),
        (195, 220, 210),
        (225, 230, 225),
        (170, 210, 200),
        (20, 30, 20),
        (50, 60, 40),
        (30, 50, 35),
        (65, 70, 60),
        (100, 110, 105),
        (165, 180, 180),
        (140, 140, 150),
        (185, 195, 195),
    ],
    "blue": [
        (60, 120, 190),
        (120, 170, 200),
        (120, 170, 200),
        (175, 210, 230),
        (145, 210, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        (90, 90, 140),
        (60, 60, 120),
        (110, 110, 175),
    ],
    "black": [
        (100, 100, 100),
    ],
}

def marker_detection(image, pen_color):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2] 
    thresholds = PENS_RGB[pen_color]

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])

        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))

    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])

    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])
    elif pen_color == "black":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b < t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    percentage = np.sum(mask)/np.size(mask)

    return percentage    
    
############################################################################### random_patch_roi #######################################################################################

#random_patch_roi:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *patch_number(int): number of selected patches.
#   *patch_shape(array): int array with the x and y shape of the patch.
#   *tissue_rate(float): tissue covered area per patch.
##Outputs: 
#   *ROI(array): a 2D with labeled patch.

def random_patch_roi(image, patch_number, patch_shape=[100,100], tissue_rate=.8, artifact_remotion=True, artifact_percentage_umbral=.1, seed=None):
  np.random.seed(seed)
    
  tissue_mask = tissue_segmentation(image)

  x_size, y_size = tissue_mask.shape
  ROI = np.zeros([x_size, y_size])
  
  x_corners, y_corners = np.random.randint(0,x_size-(patch_shape[0]+1), size = 1000*patch_number), np.random.randint(0,y_size-(patch_shape[0]+1), size = 1000*patch_number)
  founded_patch = 0
  executed = 0
  while founded_patch < patch_number:
    x_corner, y_corner = x_corners[executed], y_corners[executed]
    
    original_patch = image[x_corner: x_corner+patch_shape[0],
                           y_corner: y_corner+patch_shape[1]]
    patch = tissue_mask[x_corner: x_corner+patch_shape[0],
                        y_corner: y_corner+patch_shape[1]]
    ROI_patch = ROI[x_corner: x_corner+patch_shape[0],
                    y_corner: y_corner+patch_shape[1]]

    patch_tissue_rate = np.sum(patch)/(patch_shape[0]*patch_shape[1])

    if artifact_remotion == True:
        marker_precentage = np.sum([marker_detection(original_patch, color) for color in ['red', 'green', 'blue', 'black']])
    else:
        marker_precentage = 0
        
    if (patch_tissue_rate >= tissue_rate) and (np.max(ROI_patch)==0) and (marker_precentage<artifact_percentage_umbral):
      founded_patch = 1 + founded_patch

      ROI[x_corner: x_corner+patch_shape[0],
          y_corner: y_corner+patch_shape[1]] = founded_patch
    
    if executed >= (len(x_corners)-1):
        executed = 0
        tissue_rate = tissue_rate - 0.05
    
    executed = executed + 1
    
  
  return(ROI)
  
############################################################################### guided_patch_roi #######################################################################################

#guided_patch_roi:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *patch_number(int): number of selected patches.
#   *relevance_mask(array): a 2D with the relevance of each pixel.
#   *patch_shape(array): int array with the x and y shape of the patch.
##Outputs: 
#   *ROI(array): a 2D with labeled patch.

def guided_patch_roi(image, patch_number, relevance_mask, patch_shape=[100,100], relevance_factor = .9, artifact_remotion=True, artifact_percentage_umbral=.1):
                  
  tissue_mask = tissue_segmentation(image)
  tissue_relevance = tissue_mask*relevance_mask

  x_size, y_size = relevance_mask.shape
  ROI = np.zeros([x_size, y_size])
 
  founded_patch = 0
  executed = 0 
  iterations = 0

  while founded_patch < patch_number:
    X,Y = np.where(tissue_relevance == np.max(tissue_relevance))
    for x, y in zip(X,Y):
      x_corner, y_corner = x - int(patch_shape[0]/2), y - int(patch_shape[1]/2)
      x_corner, y_corner = int(x_corner * (x_corner>0)), int(y_corner * (y_corner>0))

      original_patch = image[x_corner: x_corner+patch_shape[0],
                            y_corner: y_corner+patch_shape[1]]

      ROI_patch = ROI[x_corner: x_corner+patch_shape[0],
                      y_corner: y_corner+patch_shape[1]]

      relevance_patch = tissue_relevance[x_corner: x_corner+patch_shape[0],
                                         y_corner: y_corner+patch_shape[1]]

      if artifact_remotion == True:
          marker_precentage = np.sum([marker_detection(original_patch, color) for color in ['red', 'green', 'blue', 'black']])
      else:
          marker_precentage = 0
      
      patch_max_relevance = np.max(relevance_patch)*relevance_factor
      patch_mean_relevance = np.mean(relevance_patch)
      ROI_label = np.max(ROI_patch)

      if (patch_mean_relevance >= patch_max_relevance) and (ROI_label==0) and (marker_precentage <= artifact_percentage_umbral):
        founded_patch = founded_patch + 1

        ROI[x_corner: x_corner+patch_shape[0],
          y_corner: y_corner+patch_shape[1]] = founded_patch
      
      else:
        tissue_relevance[x_corner: x_corner+patch_shape[0], y_corner: y_corner+patch_shape[1]] = 0

  return ROI
  
################################################################################# patch_extraction #######################################################################################

#patch_extraction:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *patch_number(int): number of selected patches.
#   *relevance_mask(array): a 2D with the relevance of each pixel.
#   *patch_shape(array): int array with the x and y shape of the patch.
##Outputs: 
#   *ROI(array): a 2D with labeled patch.

def patch_extraction(slide, src_index, dst_index, mask, magnitude_factor = 1):
  size_by_mag = list(slide.level_dimensions)

  base2src_rate = size_by_mag[0][0]/size_by_mag[src_index][0]
  dst2src_rate = size_by_mag[dst_index][0]/size_by_mag[src_index][0]
    
  x,y = np.where(mask==True)

  x_range = np.array([np.min(x), np.max(x)])
  y_range = np.array([np.min(y), np.max(y)])

  x_range_mapped = x_range * base2src_rate
  y_range_mapped = y_range * base2src_rate

  range_dst_mapped = [(x_range[1] - x_range[0])*dst2src_rate, (y_range[1] - y_range[0])*dst2src_rate]
  
  optimized_patch = np.array(slide.read_region((int(y_range_mapped[0]), int(x_range_mapped[0])),
                                                dst_index, 
                                                (int(range_dst_mapped[1]), int(range_dst_mapped[0]))).convert('RGB'))
  optimized_patch = optimized_patch[::magnitude_factor,::magnitude_factor,:]
  
  return(optimized_patch)
  
################################################################################# patch_extractions #######################################################################################

#patch_extractions:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *patch_number(int): number of selected patches.
#   *relevance_mask(array): a 2D with the relevance of each pixel.
#   *patch_shape(array): int array with the x and y shape of the patch.
##Outputs: 
#   *ROI(array): a 2D with labeled patch.

def patch_extractions(slide, src_index, dst_index, mask, magnitude_factor = 1, workers = None, verbose = True):
  
  labels = np.unique(mask)[1:]
  
  if workers == None:
      workers = len(labels)
  executor = ThreadPoolExecutor(max_workers = workers)

  process = []
  for label in labels:
    label_mask = (mask==label)
    patch_process = executor.submit(patch_extraction, slide, src_index, dst_index, label_mask, magnitude_factor = magnitude_factor)
    process.append(patch_process)
    
  extracted_patches = 0 
  while False in [patch_process.done() for patch_process in process]:
    
    if (verbose == True) and (np.sum([patch_process.done() for patch_process in process]) !=  extracted_patches):
      execution_flags = np.array([patch_process.done() for patch_process in process])
      sys.stdout.write(f"\r Waiting ({100 * np.sum(execution_flags)/len(execution_flags)})% ...")
      sys.stdout.flush()
    
    extracted_patches = np.sum([patch_process.done() for patch_process in process])

  patches = []
  for patch_process in process:
    patches.append(patch_process.result())
  
  return (np.array(patches))
  
############################################################################### patch_assembly #####################################################################################

#patch_assembly:
#Performe blue_ratio filter
##Inputs: 
#   *image(array): an RGB image.
#   *patch_number(int): number of selected patches.
#   *relevance_mask(array): a 2D with the relevance of each pixel.
#   *patch_shape(array): int array with the x and y shape of the patch.
##Outputs: 
#   *ROI(array): a 2D with labeled patch.

def patch_assembly(patch_list, assambly_size=None):
  n_patches = patch_list.shape[0]
  n_chanels = patch_list.shape[3]
  patch_x_size, patch_y_size =  patch_list.shape[1], patch_list.shape[2]
  
  if assambly_size == None:
    edge_size = np.ceil(n_patches**(1/2))
    assambly = np.uint8(np.zeros([int(edge_size*patch_x_size), int(edge_size*patch_y_size), n_chanels]))
  else:
    assambly = np.uint8(np.zeros([assambly_size[0], assambly_size[1], n_chanels]))
  
  row_index = 0
  col_index = 0
  for i in range(n_patches):
    patch = patch_list[i]

    assambly[row_index:row_index+patch_x_size, col_index:col_index+patch_y_size] = patch
    col_index = col_index + patch_y_size
    
    if col_index >= assambly.shape[1]:
      row_index = row_index + patch_x_size
      col_index = 0

  return(assambly)