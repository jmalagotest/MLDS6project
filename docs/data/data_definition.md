# Definición de los datos

## Origen de los datos

- [ ] La fuente de datos es un reto de kaggle llamado [Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) el cual es necesario inscribirse para poder realizar la descarga de las imagenes y los archivos csv donde estan clasificadas.

## Especificación de los scripts para la carga de datos

- [ ] Para la importación de los datos se utiliza el script ubicado en este repositorio en la ruta [scripts/data_acquisition/main.py], este realiza el clonado en local del repositorio de git del proyecto y luego realiza la descarga de los archivos de imagenes de un repositorio de Google Drive

## Referencias a rutas o bases de datos origen y destino

- [ ] Luego de la carga de datos por el script [scripts/data_acquisition/main.py] se realiza descarga en la ruta [src/database/cvpr2020-plant-pathology/data/train.csv'] en donde se encuentra la clasificacion de las imagenes.

### Rutas de origen de datos

- [ ] los archivos del repositorio se encuentran en la ruta [src/database/cvpr2020-plant-pathology] divididos en dos carpetas data e images
- [ ] En la ruta [src/database/cvpr2020-plant-pathology/data] se encuentra el archivo train.csv, sample_submission.csv  y test.csv que contienen la clasificacion de las imagenes, donde muestra los siguientes campos 'image_id', 'healthy', 'multiple_diseases', 'rust' y 'scab'. En la carpeta images se encuentran 3645 archivos de diferentes resoluciones.
- [ ] Luego de realizar la descarga, se realiza un preprocesamiento en el cual se realiza la disminucion de la resolucion de las imagenes y su estandarizacion para poder realizar el entrenamiento necesario, se utiliza el archivo test.csv para tomar las imagenes de prueba y poder entregar el modelo. luego de esto se realiza la utilizacion de entrenamiento sobre los datos del archivo sample_submission y luego con train.

### Base de datos de destino

- [ ] Luego de ser entrenado el modelo se guarda en un archivo .h5 y se adicionan los cambios de hiperparametros en dvc. Los scripts de realizacion de las transformaciones y documentacion del proyecto se realizan a traves de git, creando un branch el cual luego se genera un pullrequest con los cambios respectivos a la rama master.
- [ ] El modelo entrenado se utiliza a traves de una solicitud por medio de un servicio REST API a traves de fastAPI.
- [ ] El proceso consiste en realizar el envío de una solicitud POST por medio de un programa similar a postman o luego de subir el microservicio utilizar la ruta http://localhost:8000/docs y debe retornar si la imagen es de una hoja sana o tipo de enfermedad que tiene.
