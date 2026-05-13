import cv2
import numpy as np
import os
import json
from datetime import datetime
from detectores import obtener_descriptores
from matriz_rotacion import (
    calcular_matriz_transformacion,
    rotar_y_combinar,
    validar_matriz_no_extrema
)
from detectores import obtener_descriptores_img

def obtener_imagenes_carpeta(carpeta):  
    imagenes = []
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    for archivo in sorted(os.listdir(carpeta)):
        ext = os.path.splitext(archivo)[1].lower()
        if ext in extensiones:
            imagenes.append(os.path.join(carpeta, archivo))
    
    return imagenes


def calcular_distancia_euclidiana(desc1, desc2):
    if desc1 is None or desc2 is None:
        return float('inf')
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    if desc1.ndim == 1:
        return np.linalg.norm(desc1 - desc2)
    diferencias = desc1 - desc2
    distancias = np.sqrt(np.sum(diferencias**2, axis=1))
    return np.min(distancias)


def encontrar_matches(kps1, descs1, kps2, descs2, config):
    if kps1 is None or kps2 is None or descs1 is None or descs2 is None:
        return []
    
    detector_type = config.get('detector', 'orb').lower()
    if detector_type in ['sift', 'surf']:
        norm = cv2.NORM_L2
    else:
        norm = cv2.NORM_HAMMING
        
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(descs1, descs2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < config['max_distance'] * n.distance:
            good_matches.append(m)
    
    return good_matches

def intentar_stitching(img_path, img_central, kps_central, descs_central, config, diccionario_imagenes):
    img_nueva = diccionario_imagenes.get(img_path)
    if img_nueva is None:
        return None, None
    
    kps_nueva, descs_nueva = obtener_descriptores_img(img_nueva, config)
    
    if kps_nueva is None or descs_nueva is None:
        return None, None
    
    matches_filtrados = encontrar_matches(kps_central, descs_central, kps_nueva, descs_nueva, config)
    
    if len(matches_filtrados) < config['min_matches']:
        return None, None
    
    if len(matches_filtrados) < 4:
        return None, None
    mejores_matches = matches_filtrados
    
    matriz, mask, _ = calcular_matriz_transformacion(mejores_matches, kps_central, kps_nueva)
    
    if matriz is None:
        return None, None
    
    if not validar_matriz_no_extrema(matriz, config):
        print(f"  Matriz de transformación muy extrema, saltando...")
        return None, None
    
    img_combinada = rotar_y_combinar(img_central, img_nueva, matriz, config)
    return img_combinada, len(matches_filtrados)

def stitch_secuencial(lista_imagenes, diccionario_imagenes, config):
    if len(lista_imagenes) == 0:
        print("No hay imágenes para procesar")
        return None
    
    print(f"Iniciando stitching con {len(lista_imagenes)} imágenes")
    
    idx_central = 0
    
    img_central = diccionario_imagenes[lista_imagenes[idx_central]]
    kps_central, descs_central = obtener_descriptores_img(img_central, config)
    
    print(f"Imagen central: {lista_imagenes[idx_central]}")
    print(f"Descriptores centro: {len(descs_central) if descs_central is not None else 0}")
    
    idx_siguiente = idx_central + 1
    
    while idx_siguiente < len(lista_imagenes):
        print(f"\nProcesando imagen {idx_siguiente + 1}/{len(lista_imagenes)}: {lista_imagenes[idx_siguiente]}")
        
        resultado = intentar_stitching(
            lista_imagenes[idx_siguiente],
            img_central,
            kps_central,
            descs_central,
            config,
            diccionario_imagenes
        )
        
        img_combinada, num_matches = resultado
        
        if img_combinada is not None:
            print(f"  Matches encontrados: {num_matches}")
            
            img_central = img_combinada
            kps_central, descs_central = obtener_descriptores_img(img_central, config)
            
            print(f"  Stitching exitoso. Imagen combinada actualizada.")
        else:
            print(f"  No se pudo hacer stitching con esta imagen, saltando...")
        
        idx_siguiente += 1
    
    return img_central


def buscar_nueva_imagen_central(lista_imagenes, idx_inicio, config, diccionario_imagenes):
    for i in range(idx_inicio, len(lista_imagenes)):
        img_i = diccionario_imagenes[lista_imagenes[i]]
        kps_i, descs_i = obtener_descriptores_img(img_i, config)
        
        if kps_i is None or descs_i is None:
            continue
        
        tiene_coincidencia = False
        
        for j in range(i + 1, len(lista_imagenes)):
            img_j = diccionario_imagenes[lista_imagenes[j]]
            kps_j, descs_j = obtener_descriptores_img(img_j, config)
            
            if kps_j is None or descs_j is None:
                continue
            
            matches_filtrados = encontrar_matches(kps_i, descs_i, kps_j, descs_j, config)
            
            if len(matches_filtrados) >= config['min_matches']:
                tiene_coincidencia = True
                print(f"Imagen {i} coincide con imagen {j}")
                break
        
        if tiene_coincidencia:
            return i, kps_i, descs_i
    
    return None, None, None


def guardar_resultado(img, nombre_base, carpeta):
    nombre_base_limpio = os.path.splitext(os.path.basename(nombre_base))[0]
    ruta = os.path.join(carpeta, f"{nombre_base_limpio}.png")
    cv2.imwrite(ruta, img)
    print(f"  Guardado: {ruta}")


def stitch_completo(lista_imagenes, diccionario_imagenes, config):
    if len(lista_imagenes) == 0:
        print("No hay imágenes para procesar")
        return None
    
    if len(lista_imagenes) == 1:
        return diccionario_imagenes[lista_imagenes[0]]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_resultados = f"resultados_{timestamp}"
    os.makedirs(carpeta_resultados, exist_ok=True)
    
    print(f"Iniciando stitching completo con {len(lista_imagenes)} imágenes")
    print(f"Resultados en: {carpeta_resultados}/")
    
    mejor_resultado = None
    mejor_conteo = 0
    
    for idx_central in range(len(lista_imagenes)):
        nombre_central = os.path.splitext(os.path.basename(lista_imagenes[idx_central]))[0]
        print(f"\n--- Probando imagen {idx_central + 1} como central: {nombre_central} ---")
        
        img_actual = diccionario_imagenes[lista_imagenes[idx_central]]
        kps_actual, descs_actual = obtener_descriptores_img(img_actual, config)
        
        if kps_actual is None or descs_actual is None:
            print(f"  No se pudieron obtener descriptores de imagen {idx_central}")
            continue
        
        img_resultado = img_actual
        kps_resultado = kps_actual
        descs_resultado = descs_actual
        imagenes_combinadas = 1
        
        for i in range(len(lista_imagenes)):
            if i == idx_central:
                continue
            
            print(f"\n  Intentando combinar con imagen {i + 1}: {lista_imagenes[i]}")
            resultado = intentar_stitching(
                lista_imagenes[i], img_resultado, kps_resultado,
                descs_resultado, config, diccionario_imagenes
            )
            img_combinada, num_matches = resultado
            
            if img_combinada is not None:
                print(f"    Matches: {num_matches} - Combinando!")
                img_resultado = img_combinada
                kps_resultado, descs_resultado = obtener_descriptores_img(img_resultado, config)
                imagenes_combinadas += 1
            else:
                print(f"    No hay suficientes matches, saltando")
        
        print(f"  Imagen {idx_central + 1}: combinó {imagenes_combinadas}/{len(lista_imagenes)} imágenes")
        
        nombre_archivo = f"{nombre_central}_{imagenes_combinadas}"
        guardar_resultado(img_resultado, nombre_archivo, carpeta_resultados)
        
        if imagenes_combinadas > mejor_conteo:
            mejor_conteo = imagenes_combinadas
            mejor_resultado = img_resultado.copy()
    
    if mejor_resultado is not None and mejor_conteo > 1:
        print(f"\nMejor combinación: {mejor_conteo} imágenes")
        return mejor_resultado
    
    print("No se pudo completar el stitching")
    return None


config_path = 'config.json'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    carpeta_imagenes = config.get('carpeta_entrada', 'imagenes')
    lista_imagenes = obtener_imagenes_carpeta(carpeta_imagenes)
    print(f"\nImágenes encontradas: {len(lista_imagenes)}")
    
    diccionario_imagenes = {}
    for i, img_path in enumerate(lista_imagenes):
        print(f"  {i + 1}. {img_path}")
        diccionario_imagenes[img_path] = cv2.imread(img_path)

    resultado = stitch_completo(lista_imagenes, diccionario_imagenes, config)
    if resultado is not None:
        output_path = config.get('output_path', './resultado.jpg')
        cv2.imwrite(output_path, resultado)
        print(f"\nResultado guardado en: {output_path}")
        print(f"Dimensiones finales: {resultado.shape}")
    else:
        print("\nNo se pudo generar la imagen resultante")
except Exception as e:
    print('Error al cargar la configuración')
    print(e)
    exit()