# Definición de los datos

## Origen de los datos

Los datos son descargados directos de la base de datos del TCGA usando su API GDC (Documentación disponible en: https://gdc.cancer.gov/developers/gdc-application-programming-interface-api ). 

## Especificación de los scripts para la carga de datos

- El script de carga se encuentra en: 'scripts/data_acquisition/dowload_data.ipynb'

## Referencias a rutas o bases de datos origen y destino

- imagenes (WSI): '/run/media/jmalagont/Thesis/Thesis/DataSet/TCGA-BRCA/WSI'
- Tabular: ''/run/media/jmalagont/Thesis/Thesis/DataSet/TCGA-BRCA/tabular/TCGA-BRCA_clinical.csv''

### Rutas de origen de datos

El conjunto de datos cuenta con dos partes: las imágenes (conjunto no estructurado) y la información tabular (conjunto de datos estructurados)
 
- Imágenes: Cuenta con un total de 1062 imágenes en formato .svs
- Tabular: Tabular: archivo .CSV donde se identifican las caracteristicas de cada paciente, en general se consideran relevante las columnas 'days_to_death', 'days_to_last_followup' y 'vital_status'. Las cuales contiene el tiempo hasta el evento de muerte, el tiempo hasta el evento de cura y un indicador de ambos eventos respectivamente. 

### Base de datos de destino

- Los datos resultantes son almacenados en un archivo HDF5, el cual contiene 3 datasets: representation, time y event:
    - Representation: Representación extraido de una ren prentrenada.
    - time: tiempo hasta el evento
    - event: Indicador del evento.
