import os
import sqlite3

conn = sqlite3.connect('DB/imagenes.db')
c = conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS imagenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT UNIQUE,
        imagen BLOB
    )
''')

def convertir_a_binario(ruta_imagen):
    with open(ruta_imagen, 'rb') as file:
        blob = file.read()
    return blob

directorio_imagenes = 'Images'

for nombre_imagen in os.listdir(directorio_imagenes):
    ruta_imagen = os.path.join(directorio_imagenes, nombre_imagen)

    if os.path.isfile(ruta_imagen) and nombre_imagen.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            imagen_binaria = convertir_a_binario(ruta_imagen)
            c.execute('''
                INSERT INTO imagenes (nombre, imagen) VALUES (?, ?)
            ''', (nombre_imagen, imagen_binaria))
            print(f"Imagen '{nombre_imagen}' añadida a la base de datos.")
        except sqlite3.IntegrityError:
            print(f"Imagen '{nombre_imagen}' ya existe en la base de datos. No se sobreescribirá.")

conn.commit()
conn.close()