"""MLDS_6_Fase_1.ipynb

1. Limpieza de los Datos**

"""

!pip install unidecode Pillow matplotlib

!pip install -U scikit-learn

# Librerías de utilidad para manipulación y visualización de datos.
from numbers import Number
import re
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unidecode import unidecode
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

#Importamos tensorflow
import tensorflow as tf
tf.config.run_functions_eagerly(True)
# Ignorar warnings.
import warnings
warnings.filterwarnings('ignore')

# Seleccionamos las semillas para efectos de reproducibilidad 
np.random.seed(0)
tf.random.set_seed(0)

# Versiones de las librerías usadas.
!python --version
print('NumPy', np.__version__)
print('Tensorflow', tf.__version__)

!git clone https://github.com/ant-research/cvpr2020-plant-pathology.git

"""
Carga de Datos
"""
data = pd.read_csv('/content/cvpr2020-plant-pathology/data/sample_submission.csv', delimiter=",")
data.head()

!wget https://storage.googleapis.com/kaggle-data-sets/604871/1167465/compressed/plant_processed.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com/20230503/auto/storage/goog4_request&X-Goog-Date=20230503T012947Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=ba4efec526b9e6f4f6d350139fcbe17638df88d6a663dedbc01a5f51e62297c8bf1b644cf58b2325c43249341d29cfceaf59e6fbb279c495694db2bdd9ff572c86ca5688a95eb0c5b362c036fbf28d185d755aa66f5409ff5e30ee63ac77f7f24189c815505c6ff2d9b0c38e20b08b246011e83fb3ca6ec3239e88a78f74bb1734e09d2b90725060f3d3a2232c8acd6f40bc95f2e127e26492b3238627cf501159df8c3eb8e067234b4eb44500cc45ed49f4b29775f8900772b41359e8ad726bf5652107795ee125e01dda603472d388cc45ddc4b87c783126e9be32a5551098511b0b1cbba2dacbcc891687597eaa53f3e6cd51d8dce4575a4481730378a59b
!unzip plant_processed.zip

all_images = []
labels = []
for i in range(15):
    cat_path = f"'/content/plant_processed/images"
    for im_path in os.listdir(cat_path):
        all_images.append(np.array(tf.keras.preprocessing.image.load_img(cat_path+im_path,
                                                                         target_size=(224, 224, 3))))
        labels.append(i)
X = np.array(all_images)
y = np.array(labels)

from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state = 5, stratify = y)
X_train, X_val,  y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state = 5, stratify = y_temp)

X_train.shape[0], X_val.shape[0], X_test.shape[0]

X_train_prep = tf.keras.applications.mobilenet.preprocess_input(X_train)
X_val_prep = tf.keras.applications.mobilenet.preprocess_input(X_val)
X_test_prep = tf.keras.applications.mobilenet.preprocess_input(X_test)

Y_train = tf.keras.utils.to_categorical(y_train)
Y_val = tf.keras.utils.to_categorical(y_val)
Y_test = tf.keras.utils.to_categorical(y_test)
