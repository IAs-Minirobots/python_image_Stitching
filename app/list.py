
import os
def obtener_imagenes_carpeta(carpeta):
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    imagenes = []
    
    for archivo in sorted(os.listdir(carpeta)):
        ext = os.path.splitext(archivo)[1].lower()
        if ext in extensiones:
            imagenes.append(os.path.join(carpeta, archivo))
    
    return imagenes

print(obtener_imagenes_carpeta('imagenes'))