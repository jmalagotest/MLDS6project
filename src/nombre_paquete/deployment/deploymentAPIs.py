from fastapi import FastAPI, UploadFile, File
import tempfile
import json

import numpy as np
from tensorflow.keras import applications, layers, models

import sys
sys.path.append('/home/jmalagont/Documentos/MLDS6project/src/nombre_paquete/preprocessing')
import pyWSI as pywsi

#Build feature extractor model
def compression_CNN (backbone, shape):
  base_model = eval(f'applications.{backbone}(include_top=False, weights="imagenet", input_shape={str(shape)})')

  cnn = models.Sequential()

  cnn.add(base_model)
  cnn.add(layers.GlobalAveragePooling2D())

  return(cnn)

BB = compression_CNN ('ResNet50V2', (4000,4000,3))

#Load inference model
BestModelPath = '/home/jmalagont/Documentos/MLDS6project/src/dataset/Best model'
comp = models.load_model(f'{BestModelPath}/modelo.h5', compile=False)
comp.load_weights(f'{BestModelPath}/pesos.h5')
    
# Init App
app = FastAPI()

#Alive function
@app.get("/")
def alive():
    return ('Alive')
    
#Preprocessing function
@app.post('/preprocessing')
async def preproccessing(file: UploadFile= File(...)):
    BB.summary()

    contents = await file.read()
    
    #save the file
    temporalfile = tempfile.NamedTemporaryFile() 
    temporalfile.write(contents)
    temporalfile.seek(0)

    #extraction specs 
    n_patches = 64
    patch_size = 500
    seed = 1997

    #Open slide
    slide = pywsi.svs_read(temporalfile.name)
    print('Slide opened')

    # Find optical ampf
    slide_properties = dict(slide.properties)
    maximum_mag = slide_properties['aperio.AppMag']
    print('Find optical amplitude')

    #Stablish reduction factor
    if maximum_mag == '40':
        reduction_factor = 2
    elif maximum_mag == '20':
        reduction_factor = 1
    else:
        reduction_factor = None
    print('Find reduction factor')

    #Establish relative patch size
    shapes = list(slide.level_dimensions)
    relative_patch_size = int((patch_size * reduction_factor) * (shapes[-1][0]/shapes[0][0])) + 2
    print('Found patch size')

    #ROI finder
    minimum_image = np.array(slide.read_region((0, 0), (len(shapes)-1), shapes[-1]).convert('RGB')) 
    ROI = pywsi.random_patch_roi(minimum_image, n_patches, [relative_patch_size, relative_patch_size], tissue_rate=.95, artifact_remotion=True, artifact_percentage_umbral=.1, seed=1997)
    print('Extract ROI')

    #Patch extarctor
    patches = pywsi.patch_extractions(slide, (len(shapes)-1), 0, ROI, magnitude_factor = reduction_factor, workers = None, verbose = False)
    patches = patches [:n_patches,:patch_size, :patch_size, :]
    print('Extract patches')

    #Buil assambky image
    assambly = pywsi.patch_assembly(patches, assambly_size=None)
    print('End')

    #Embbeding extraction
    embbeding = BB(np.array([assambly])).numpy()[0]
    
    return {"Image features": embbeding.tolist()}

#Preprocessing function
@app.post('/infer')
async def infer(file: UploadFile= File(...)):
    comp.summary()

    contents = await file.read()
    features = json.loads(contents)
    features_vector = features['Image features']
    estimated_hazard = comp(np.array([features_vector])).numpy()[0]

    return {"Hazards": estimated_hazard.tolist()}