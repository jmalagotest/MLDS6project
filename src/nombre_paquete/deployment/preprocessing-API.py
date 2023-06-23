from fastapi import FastAPI, UploadFile, File
import tempfile

import numpy as np
import cv2
import sys
sys.path.append('/home/jmalagont/Documentos/MLDS6project/src/nombre_paquete/preprocessing')
import pyWSI as pywsi

# Init App
app = FastAPI()

#Alive function
@app.get("/")
def alive():
    return ('Alive')
    

@app.post('/preprocessing')
async def preproccessing(file: UploadFile= File(...)):

    contents = await file.read()
    
    #save the file
    temporalfile = tempfile.NamedTemporaryFile() 
    temporalfile.write(contents)
    temporalfile.seek(0)


    

    n_patches = 64
    patch_size = 500
    seed = 1997

    slide = pywsi.svs_read(temporalfile.name)

    slide_properties = dict(slide.properties)
    maximum_mag = slide_properties['aperio.AppMag']

    if maximum_mag == '40':
        reduction_factor = 2
    elif maximum_mag == '20':
        reduction_factor = 1
    else:
        reduction_factor = None

    shapes = list(slide.level_dimensions)
    relative_patch_size = int((patch_size * reduction_factor) * (shapes[-1][0]/shapes[0][0])) + 2
    
    minimum_image = np.array(slide.read_region((0, 0), (len(shapes)-1), shapes[-1]).convert('RGB')) 
    ROI = pywsi.random_patch_roi(minimum_image, n_patches, [relative_patch_size, relative_patch_size], tissue_rate=.95, artifact_remotion=True, artifact_percentage_umbral=.1, seed=1997)
    
    patches = pywsi.patch_extractions(slide, (len(shapes)-1), 0, ROI, magnitude_factor = reduction_factor, workers = None, verbose = False)
    patches = patches [:n_patches,:patch_size, :patch_size, :]

    assambly = pywsi.patch_assembly(patches, assambly_size=None)
    cv2.imwrite('savedimage.jpeg', assambly)   

    return {"filename": [int(np.min(assambly)), int(np.max(assambly))]}