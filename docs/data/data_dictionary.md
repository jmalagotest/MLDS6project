# Diccionario de datos
 
## Base de datos 1 train.csv

Esta base de datos contienne la clasificacion de las plantas y su estado, el nombre va enlazado con el nombre de la imagen en la carpeta de images


| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| image_id | Id que enlaza con la imagen de la carpeta | Texto | A-Z a-z 0-9 | train.csv |
| healthy | Clasificacion planta sana o enferma | Boolean | False/True 0/1 | train.csv |
| multiple_diseases | Clasificacion multiples enfermedades | Boolean | False/True 0/1  | train.csv |
| rust | Clasificacion roya como enfermedad | Boolean | False/True 0/1  | train.csv |
| scab | Clasificacion sarna como enfermedad | Boolean | False/True 0/1  | train.csv  |

- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.

## Base de datos 2 sample_submission.csv

Esta base de datos contiene los ejemplos principales de tipo de enfermedad clasificados.

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| image_id | Id que enlaza con la imagen de la carpeta | Texto | A-Z a-z 0-9 | train.csv |
| healthy | Clasificacion planta sana o enferma | Boolean | False/True 0/1 | train.csv |
| multiple_diseases | Clasificacion multiples enfermedades | Boolean | False/True 0/1  | train.csv |
| rust | Clasificacion roya como enfermedad | Boolean | False/True 0/1  | train.csv |
| scab | Clasificacion sarna como enfermedad | Boolean | False/True 0/1  | train.csv  |

- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.

## Base de datos 3 test.csv

Esta base de datos contiene las imagenes que optan como uso para test del modelo.

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| image_id | Id que enlaza con la imagen de la carpeta | Texto | A-Z a-z 0-9 | train.csv |

- **Variable**: nombre de la variable.
- **Descripción**: breve descripción de la variable.
- **Tipo de dato**: tipo de dato que contiene la variable.
- **Rango/Valores posibles**: rango o valores que puede tomar la variable.
- **Fuente de datos**: fuente de los datos de la variable.
