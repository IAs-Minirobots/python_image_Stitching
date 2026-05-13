# Image Stitcher

Herramienta de stitching panorámico que combina múltiples imágenes superpuestas en una sola imagen panorámica.

## Características

- **3 detectores de características**: SIFT, ORB, AKAZE, BRISK
- **Validación de homografía**: Rechaza transformaciones con rotación, escala o perspectiva extremas
- **Búsqueda exhaustiva**: Prueba todas las imágenes como central y elige la mejor combinación
- **Múltiples modos de combinación**: average, blend, multiply, overlay, central
- **Exporta todos los resultados**: Guarda cada combinación intentada en `resultados_{timestamp}/`

## Estructura

```
├── app/
│   ├── main.py              # Orquestador del stitching
│   ├── detectores.py         # Extracción de descriptores (SIFT, ORB, etc.)
│   └── matriz_rotacion.py    # Homografía, validación y combinación
├── config.json               # Configuración del detector y parámetros
├── imagenes/                 # Carpeta de entrada (configurable)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Uso

```bash
# Local
pip install -r requirements.txt
python app/main.py

# Docker
docker compose up --build
```

## Configuración (`config.json`)

| Parámetro | Descripción | Default |
|---|---|---|
| `detector` | Algoritmo: sift, orb, akaze, brisk | sift |
| `max_descriptors` | Límite de descriptores | 6000 |
| `max_distance` | Umbral de Lowe para matches (0.0-1.0) | 0.2 |
| `min_matches` | Mínimo de coincidencias para stitching | 10 |
| `max_rotation_angle` | Rotación máxima permitida (grados) | 120 |
| `max_scale_change` | Escala máxima permitida | 5.0 |
| `min_scale_change` | Escala mínima permitida | 0.1 |
| `max_perspective` | Distorsión perspectiva máxima | 0.1 |
| `combination_mode` | average, blend, multiply, overlay, central | average |
| `carpeta_entrada` | Carpeta con imágenes de entrada | imagenes |

### Modos de combinación

- **average**: Promedia píxeles donde se superponen
- **blend**: Mezcla con transparencia (alpha 0.5)
- **multiply**: Multiplica píxeles con función sigmoide
- **overlay**: Zonas superpuestas se promedian
- **central**: Prioriza la imagen central, solo añade píxeles donde la central está vacía

## Algoritmo

1. Toma cada imagen como central y la combina secuencialmente con las demás
2. Para cada par, extrae descriptores, encuentra matches y calcula homografía
3. Valida que la transformación no sea extrema (rotación, escala, perspectiva)
4. Si es válida, combina las imágenes y continúa con la siguiente
5. Guarda todos los resultados en `resultados_{timestamp}/`
6. Al final, retorna la combinación con más imágenes
