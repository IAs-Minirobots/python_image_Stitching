import cv2
import numpy as np

class DescriptorExtractor:
    def __init__(self, detector_type, max_descriptors):
        self.detector_type = detector_type.lower()
        self.max_descriptors = max_descriptors
        self.detector = self._create_detector()

    def _create_detector(self):
        if self.detector_type == 'sift':
            return cv2.SIFT_create(nfeatures=self.max_descriptors)
        elif self.detector_type == 'surf':
            try:
                return cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=True)
            except AttributeError:
                raise ValueError("SURF no está disponible. Requiere opencv-contrib-python compilado con OPENCV_ENABLE_NONFREE=ON. Usa 'sift', 'orb', 'akaze' o 'brisk'.")
            except cv2.error as e:
                raise ValueError(f"Error al cargar SURF (probablemente necesites una versión con OPENCV_ENABLE_NONFREE): {e}")
        elif self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=self.max_descriptors)
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        elif self.detector_type == 'brisk':
            return cv2.BRISK_create()
        else:
            raise ValueError(f"Detector no soportado: {self.detector_type}. Opciones: sift, surf, orb, akaze, brisk")

    def detect_and_compute(self, image):
        if image is None:
            return None, None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors


def obtener_descriptores_img(imagen, config):
    if imagen is None:
        return None, None
    
    extractor = DescriptorExtractor(
        detector_type=config['detector'],
        max_descriptors=config['max_descriptors']
    )
    
    return extractor.detect_and_compute(imagen)

def obtener_descriptores(imagen_path, config):
    imagen = cv2.imread(imagen_path)
    return obtener_descriptores_img(imagen, config)