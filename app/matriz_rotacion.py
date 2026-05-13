import cv2
import numpy as np

def calcular_matriz_transformacion(matches, kps1, kps2):
    if len(matches) < 4:
        return None, None, None
    
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    return M, mask, pts1


def validar_matriz_no_extrema(M, config):
    if M is None:
        return False
    
    max_rotation = np.radians(config.get('max_rotation_angle', 30))
    max_scale = config.get('max_scale_change', 2.0)
    min_scale = config.get('min_scale_change', 0.5)
    max_perspective = config.get('max_perspective', 0.02)

    angulo = abs(np.arctan2(M[1, 0], M[0, 0]))
    
    if angulo > max_rotation:
        return False
    
    scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
    scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
    
    if scale_x > max_scale or scale_x < min_scale:
        return False
    if scale_y > max_scale or scale_y < min_scale:
        return False
    
    if abs(M[2, 0]) > max_perspective or abs(M[2, 1]) > max_perspective:
        return False
    
    return True


def filtrar_matches_por_calidad(matches, kps1, kps2, max_distance, quality_threshold):
    good_matches = []
    
    for match in matches:
        distancia = match.distance
        if distancia > max_distance:
            continue
        
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        
        if query_idx >= len(kps1) or train_idx >= len(kps2):
            continue
        
        pt1 = kps1[query_idx].pt
        pt2 = kps2[train_idx].pt
        
        all_distances = []
        for other_match in matches:
            if other_match == match:
                continue
            other_pt1 = kps1[other_match.queryIdx].pt
            other_pt2 = kps2[other_match.trainIdx].pt
            d = np.sqrt((pt1[0] - other_pt1[0])**2 + (pt1[1] - other_pt1[1])**2)
            all_distances.append(d)
        
        if len(all_distances) > 0:
            avg_distance = np.mean(all_distances)
            if avg_distance > 0:
                ratio = distancia / avg_distance
                if ratio < quality_threshold:
                    good_matches.append((match, distancia))
        
    good_matches.sort(key=lambda x: x[1])
    
    return [m[0] for m in good_matches]


def obtener_mejores_matches(matches, kps1, kps2, config, num_mejores=3):
    max_dist = config.get('max_distance', 0.7)
    quality_thresh = config.get('quality_threshold', 0.7)
    
    filtrados = filtrar_matches_por_calidad(matches, kps1, kps2, max_dist, quality_thresh)
    
    if len(filtrados) < num_mejores:
        return filtrados
    
    return filtrados[:num_mejores]


def rotar_y_combinar(img_central, img_nueva, matriz, config):
    h1, w1 = img_central.shape[:2]
    h2, w2 = img_nueva.shape[:2]
    
    esquinas_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    esquinas_img2_trans = cv2.perspectiveTransform(esquinas_img2, matriz)
    
    esquinas_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    todas_esquinas = np.concatenate((esquinas_img1, esquinas_img2_trans), axis=0)
    
    [x_min, y_min] = np.int32(todas_esquinas.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(todas_esquinas.max(axis=0).ravel() + 0.5)
    
    t_x = -x_min if x_min < 0 else 0
    t_y = -y_min if y_min < 0 else 0
    
    matriz_traslacion = np.array([
        [1, 0, t_x],
        [0, 1, t_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    matriz_final = matriz_traslacion.dot(matriz)
    
    new_w = x_max - x_min
    new_h = y_max - y_min
    
    img2_warped = cv2.warpPerspective(img_nueva, matriz_final, (new_w, new_h))
    
    img1_warped = np.zeros((new_h, new_w, 3), dtype=img_central.dtype)
    img1_warped[t_y:t_y+h1, t_x:t_x+w1] = img_central
    
    return aplicar_combinacion(config, img1_warped, img2_warped)


def combinar_imagenes(img1, img2, mask1=None, mask2=None, modo='blend'):
    if modo == 'average':
        return promediar_pixeles(img1, img2)
    elif modo == 'multiply':
        return multiplicar_sigmoide(img1, img2)
    elif modo == 'blend':
        return blend_images(img1, img2)
    elif modo == 'overlay':
        return overlay_images(img1, img2)
    else:
        return promediar_pixeles(img1, img2)


def promediar_pixeles(img1, img2):
    if img1 is None or img2 is None:
        return img1 if img2 is None else img2
    
    mask1 = (img1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (img2.sum(axis=2) > 0).astype(np.float32)
    mask_combined = mask1 + mask2
    mask_combined[mask_combined == 0] = 1
    
    resultado = (img1.astype(np.float32) + img2.astype(np.float32)) / mask_combined[:, :, np.newaxis]
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    
    return resultado


def multiplicar_sigmoide(img1, img2):
    def sigmoide(x):
        return 1 / (1 + np.exp(-(x - 127) / 30))
    
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0
    
    resultado = img1_float * img2_float
    
    resultado = np.clip(resultado * 255, 0, 255).astype(np.uint8)
    
    return resultado


def blend_images(img1, img2, alpha=0.5):
    # En rotar_y_combinar las imágenes ya tienen exactamente el mismo tamaño
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def overlay_images(img1, img2):
    # En rotar_y_combinar las imágenes ya tienen exactamente el mismo tamaño
    h, w = img1.shape[:2]
    resultado = np.zeros((h, w, 3), dtype=np.uint8)
    
    mask1 = (img1.sum(axis=2) > 0)
    mask2 = (img2.sum(axis=2) > 0)
    
    mask_overlap = mask1 & mask2
    mask_only1 = mask1 & ~mask_overlap
    mask_only2 = mask2 & ~mask_overlap
    
    resultado[mask_only1] = img1[mask_only1]
    resultado[mask_only2] = img2[mask_only2]
    
    if np.any(mask_overlap):
        overlap = np.zeros((h, w, 3), dtype=np.float32)
        overlap[mask_overlap] = (img1[mask_overlap].astype(np.float32) + img2[mask_overlap].astype(np.float32)) / 2
        resultado[mask_overlap] = overlap[mask_overlap].astype(np.uint8)
    
    return resultado


def priorizar_central(img_central, img_nueva):
    h, w = img_central.shape[:2]
    resultado = img_central.copy()
    
    mask_vacia = (img_central.sum(axis=2) == 0)
    resultado[mask_vacia] = img_nueva[mask_vacia]
    
    return resultado


def aplicar_combinacion(config, img1, img2):
    modo = config.get('blend_mode', 'average')
    combination_mode = config.get('combination_mode', 'blend')
    
    if combination_mode == 'blend':
        return blend_images(img1, img2)
    elif combination_mode == 'average':
        return promediar_pixeles(img1, img2)
    elif combination_mode == 'multiply':
        return multiplicar_sigmoide(img1, img2)
    elif combination_mode == 'overlay':
        return overlay_images(img1, img2)
    elif combination_mode == 'central':
        return priorizar_central(img1, img2)
    else:
        return promediar_pixeles(img1, img2)