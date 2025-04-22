import os
import sqlite3
import cv2
import numpy as np

protoFile = "model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

keypointsMapping = [
    "Head", "Neck", "R-Sho", "R-Elb", "R-Wr", "L-Sho", "L-Elb", "L-Wr",
    "R-Hip", "R-Knee", "R-Ank", "L-Hip", "L-Knee", "L-Ank", "Waist"
]

POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
    [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]
]

def get_skeleton_points(frame, draw=False, prob_threshold=0.2):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H, W = output.shape[2], output.shape[3]
    points = []
    
    for i in range(len(keypointsMapping) - 1):  
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        x = int((frameWidth * point[0]) / W)
        y = int((frameHeight * point[1]) / H)

        if prob > prob_threshold:
            points.append((x, y))
            if draw:
                circle_radius = max(1, int(min(frameWidth, frameHeight) * 0.02))
                cv2.circle(frame, (x, y), circle_radius, (0, 0, 255), thickness=-1,
                lineType=cv2.FILLED) 
        else:
            points.append(None)

    if points[8] and points[11]:
        waist_x = (points[8][0] + points[11][0]) // 2
        waist_y = (points[8][1] + points[11][1]) // 2
        points.append((waist_x, waist_y))
    else:
        points.append(None)

    if draw:
        for pair in POSE_PAIRS:
            partA, partB = pair
            if points[partA] and points[partB]:
                line_thickness = max(1, int(min(frameWidth, frameHeight) * 0.005))
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), thickness=line_thickness)
    
    return points

def crop_image_around_skeleton(frame, points, padding_percent=15):
    # Filtrar los puntos clave válidos
    valid_points = [point for point in points if point is not None]
    
    if not valid_points:
        return frame  # Si no hay puntos válidos, devolver la imagen original
    
    # Encontrar los límites del recorte
    x_coords, y_coords = zip(*valid_points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calcular el padding en función del tamaño de la imagen
    padding_x = int(frame.shape[1] * padding_percent / 100)
    padding_y = int(frame.shape[0] * padding_percent / 100)
    
    # Añadir padding
    x_min = max(0, x_min - padding_x)
    x_max = min(frame.shape[1], x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(frame.shape[0], y_max + padding_y)
    
    # Recortar la imagen
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    
    return cropped_frame

def normalize_points(points):
    valid_points = [point for point in points if point is not None]
    
    if not valid_points:
        return points
    
    x_coords, y_coords = zip(*valid_points)
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    
    normalized_points = []
    
    for point in points:
        if point is not None:
            normalized_x = (point[0] - x_min) / (max(x_coords) - x_min)
            normalized_y = (point[1] - y_min) / (max(y_coords) - y_min)
            normalized_points.append((normalized_x, normalized_y))
        else:
            normalized_points.append(None)
    
    return normalized_points

def calculate_vector(p1, p2):
    return np.array(p2) - np.array(p1)

def compare_poses_using_vectors(points1, points2):
    vectors1 = []
    vectors2 = []
    
    vector_pairs_indices = [
        (0, 1),   # Head to Neck
        (2, 3),   # R-Sho to R-Elb
        (3, 4),   # R-Elb to R-Wr
        (8, 9),   # R-Hip to R-Knee
        (9, 10),  # R-Knee to R-Ank
        (5, 6),   # L-Sho to L-Elb
        (6, 7),   # L-Elb to L-Wr
        (11, 12), # L-Hip to L-Knee
        (12, 13)  # L-Knee to L-Ank
    ]
    
    for idx_pair in vector_pairs_indices:
        partA, partB = idx_pair
        if points1[partA] and points1[partB]:
            vector1 = calculate_vector(points1[partA], points1[partB])
            vectors1.append(vector1)
        if points2[partA] and points2[partB]:
            vector2 = calculate_vector(points2[partA], points2[partB])
            vectors2.append(vector2)

    if len(vectors1) != len(vectors2):
        print ((len(vectors1),len(vectors2)))
        return float('inf')  # No valid vectors to compare

    total_difference = sum(np.linalg.norm(v1 - v2) for v1, v2 in zip(vectors1, vectors2))

    return total_difference

import os
import datetime

# Obtener la fecha actual
fecha_actual = datetime.datetime.now().strftime("%m-%Y")

# Crear el directorio de resultados con la fecha actual
directorio_resultados = os.path.join("results", fecha_actual)
os.makedirs(directorio_resultados, exist_ok=True)

# Contar el número de pruebas existentes en el directorio
numero_pruebas = len([d for d in os.listdir(directorio_resultados) if os.path.isdir(os.path.join(directorio_resultados, d))]) + 1

# Crear el subdirectorio para la nueva prueba
directorio_prueba = os.path.join(directorio_resultados, f"prueba_{numero_pruebas}")
os.makedirs(directorio_prueba, exist_ok=True)

# Obtener la última captura realizada
captures_dir = "captures"
imagenes_capturadas = [os.path.join(captures_dir, f) for f in os.listdir(captures_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
ultima_imagen = max(imagenes_capturadas, key=os.path.getctime)

# Conectar a la base de datos SQLite
conn = sqlite3.connect('DB/imagenes.db')
c = conn.cursor()

# Obtener las imágenes de la base de datos
c.execute("SELECT nombre, imagen FROM imagenes")
imagenes_db = c.fetchall()

imagen_a_comparar = cv2.imread(ultima_imagen)

if imagen_a_comparar is None:
    print("Error al cargar la imagen a comparar.")
else:
    # Obtener los puntos clave del esqueleto para la imagen a comparar
    puntos_a_comparar = get_skeleton_points(imagen_a_comparar, draw=False)
    # Obtener imagen cropped
    cropped_frame_a_comparar = crop_image_around_skeleton(imagen_a_comparar, puntos_a_comparar)
    # Obtener los puntos clave del esqueleto para la imagen cropped
    cropped_puntos_a_comparar = get_skeleton_points(cropped_frame_a_comparar,draw=True)
    # Normalizar los puntos
    puntos_a_comparar_normalizados = normalize_points(cropped_puntos_a_comparar)

    mejor_indice = float('inf')
    mejor_nombre = None
    mejor_frame = None
    indices_coincidencia = []

    for nombre, imagen_blob in imagenes_db:
        imagen_db = cv2.imdecode(np.frombuffer(imagen_blob, np.uint8), cv2.IMREAD_COLOR)
        if imagen_db is not None:
            # Obtener los puntos clave del esqueleto para la imagen de la base de datos
            puntos_db = get_skeleton_points(imagen_db, draw=False)
            # Obtener imagen cropped
            cropped_frame_db = crop_image_around_skeleton(imagen_db, puntos_db)
            # Obtener los puntos clave del esqueleto para la imagen cropped
            cropped_puntos_db = get_skeleton_points(cropped_frame_db,draw=True)
            # Normalizar los puntos
            puntos_db_normalizados = normalize_points(cropped_puntos_db)

            cv2.imwrite(f"test/{nombre}", cropped_frame_db)

            # Comparar los puntos normalizados
            similarity_index = compare_poses_using_vectors(puntos_a_comparar_normalizados, puntos_db_normalizados)

            indices_coincidencia.append((nombre, similarity_index))

            if similarity_index < mejor_indice:
                mejor_indice = similarity_index
                mejor_nombre = nombre
                mejor_frame = imagen_db
                mejor_frame_cropped = cropped_frame_db

    if mejor_nombre:
        mejor_nombre_sin_extension = mejor_nombre.replace(".jpg", "")
        print(f"La imagen más parecida es: {mejor_nombre_sin_extension} con una distancia de {mejor_indice}")

        # Guardar las imágenes recortadas con el esqueleto dibujado
        cv2.imwrite(os.path.join(directorio_prueba, "imagen_a_comparar_con_esqueleto.jpg"), cropped_frame_a_comparar)
        cv2.imwrite(os.path.join(directorio_prueba, f"{mejor_nombre_sin_extension}_con_esqueleto.jpg"), mejor_frame_cropped)
    else:
        print("No se encontró una imagen similar en la base de datos.")

    print("\nDistancia con todas las imágenes:")
    for nombre, indice in indices_coincidencia:
        print(f"{nombre}: {indice}")

conn.close()