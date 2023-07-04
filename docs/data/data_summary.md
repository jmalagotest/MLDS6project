# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

Los datos fueron obtenidos del reto de kaggle [Plant Pathology 2020](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7), los cuales se basaban en dos partes. La primera es las 3645 imagenes(image_id) de diferentes calidades y tamaños, la segunda parte esta conformada por los archivos ddonde se encuentra la clasificacion realizada manualmente y de la cual se quiere entrenar el modelo para que detecte automaticamente de una imagen si la planta tiene enfermedad(healthy), si tiene mas de una enfermedad(multiple_diseases) y si puede clasificarla en dos posibles variables o tipos de enfermedad (rust)(scab).

| image_id | Id que enlaza con la imagen de la carpeta | Texto | A-Z a-z 0-9 | train.csv |
| healthy | Clasificacion planta sana o enferma | Boolean | False/True 0/1 | train.csv |
| multiple_diseases | Clasificacion multiples enfermedades | Boolean | False/True 0/1  | train.csv |
| rust | Clasificacion roya como enfermedad | Boolean | False/True 0/1  | train.csv |
| scab | Clasificacion sarna como enfermedad | Boolean | False/True 0/1  | train.csv  |

## Resumen de calidad de los datos

Se encuentra que las imagenes no vienen en el mismo tamaño, ni resolucion, ni calidad y no se detecta en los archivos, imagenes duplicadas. Por lo cual solo see utiliza un formateo a un mismo tamaño de imagen y de esta manera poder realizar un entrenamiento con imagenes del mismo tamaño, calidad y resolucion.

## Variable objetivo

El reto busca que se pueda clasificar cuando una planta a traves de una imagen esta sana(healthy) o enferma.

## Variables individuales

El reto viene con tres variables individuales:
 - multiple_diseases -> si la planta no esta sana(variable healthy), esta variable se utiliza para clasificar si tiene multiples enfermedades.
 - rust -> si la planta no esta sana(variable healthy), esta variable se utiliza para clasificar si tiene la enfermedad de roya.
 - scab -> si la planta no esta sana(variable healthy), esta variable se utiliza para clasificar si tiene la enfermedad de sarna.
Como tal el reto restringe cualquier transformacion a las variables.

## Ranking de variables

El ranking de variables esta preestablecido en el reto de kaggle de la siguiente manera:
- scab -> se clasifica de acuerdo a esta variable si la planta presenta la enfermedad de sarna o no.
- rust -> se clasifica de acuerdo a esta variable si la planta presenta la enfermedad de roya o no.
- multiple_diseases -> se clasifica de acuerdo a esta variable si la planta presenta las dos enfermedades o no
- healthy -> se clasifica de acuerdo a las variables explicativas si la planta presenta enfermedad/enfermedades o no.

## Relación entre variables explicativas y variable objetivo

Existe una relacion directa entre las variables explicativas y la variable objetivo, ya que si al menos una o todas las variables son positivas, significa que la planta esta enferma. En el analisis realizado en el archivo [analisis_variables.ipynb](src/plan_pathology_2020/preprocessing/analisis_variables.ipynb) sobre el archivo principal de clasificacion train.csv, muestra la correlacion que estos tienen y como la tienen.