En este fichero se explicará detalladamente el funcionamiento de la herramienta.

En primer lugar se explicarán los directorios del directorio raíz:

    · captures: Contiene las imágenes capturadas con la ejecución del código del módulo po-
    se_camera.py.

    · DB: Contiene las bases de datos con las imágenes almacenadas mediante la ejecución del
    código del módulo load_DB.py.

    · Images: Contiene las imágenes que se utilizan para almacenar en las distintas bases de datos.
    
    · model: Contiene los archivos necesarios para el funcionamiento de la red neuronal, descritos
    anteriormente.

    · results: Contiene los resultados de la ejecución del algoritmo, mostrando la imagen de inicio
    con esqueleto y la de mayor índice de coincidencia con esqueleto.

    · test: Contiene la imágenes generadas tras ejecutar el algoritmo, sobre las imágenes de la base
    de datos, para controlar que se están generando correctamente.

Antes de explicar los distintos módulos de la herramienta, se explicaran los archivos que permiten el uso de la red neuronal(dentro del directorio model):

    · pose_iter_160000.caffemodel : Este archivo contiene los pesos del modelo preentrenado para la estimación de poses. Es el resultado del entrenamiento del modelo y se utiliza para realizar predicciones sobre nuevas imágenes.

    · pose_deploy_linevec_faster_4_stages.prototxt: Este archivo define la arquitectura de la red neuronal utilizada para la estimación de poses. Especifica las capas y las conexiones entre ellas, y se utiliza junto con el archivo de pesos para realizar las predicciones.

A continuación, se explicarán los módulos que contiene la herramienta:

    · image_vector_comparator_BD.py: Contiene los scrips necesarios para utilizar el modelo de red neuronal y comparar las imágenes en función a un índice de coincidencia.

    · load_DB.py: Contiene el código necesario para la creación de una base de datos SQLite, con una tabla que contiene el ID, título y conversión en binario de la imagen. Las imágenes que se subirán a dichas tablas deben estar contenidas en el directorio Images.
    
    · pose_camera.py: Contiene el scrip necesario para acceder al dispositivo predeterminado de video del dispositivo dónde se ejecute. Además de acceder, permite realizar capturas, que se almacenan en el directorio captures.

Una vez explicados los módulos y los directorios que conforman el proyecto, se explicará paso por paso una ejecución completa de la herramienta:

    1º Hay que almacenar las imágenes con las que se compararán las capturas en una base de datos. Para ello, hay que añadir todas las imágenes en el directorio Images y ejecutar desde la terminal el módulo load_DB.py

    2º Tras almacenar las imágenes en una base de datos, hay que tomar una captura. Para tomar una captura, hay que ejecutar desde la terminal el módulo pose_camera.py, tras ejecutarlo se abrirá una ventana emergente, con la imagen que toma el dispositivo predeterminado de vídeo del dispositivo en el que se lanza, el funcionamiento del script comprende de dos teclas, con la tecla "c", se realiza una captura de pantalla y con la tecla "ESC" se cierra la ventana emergente y se detiene la ejecución.

    3º Por último, una vez tenemos las imágenes almacenadas y una captura realizada, ejecutamos desde la terminal el módulo image_vector_comparator_BD.py, con el cual se calculará automáticamente la diferencia entre la captura y las imágenes de la base de datos, generando en el directorio results la captura y la imagen más parecida con su esqueleto dibujado, generando en el directorio test una imagen con esqueleto dibujado por imagen de la base de datos y devolviendo en la terminal un resumen con la imagen más parecida y la distancia de la captura respecto a todas las imágenes de la base de datos.