Challenge: https://www.kaggle.com/c/siim-isic-melanoma-classification/

Este es el dataset que usamos: https://www.kaggle.com/sensioai/melanoma224. Contiene las imágenes del dataset original re-escaladas a 224x224 píxeles.

# Día 1

- Intro competi
- Descarga datos
- [Exploración datos](./exploracion.ipynb)
- Sample submission -> 0.5
- [Baseline Keras](./keras_baseline.ipynb): resnet50 congelada, cabeza lineal entrenada en subset 224px, adam 1e-3, batch size 64 -> 0.764
  
# Día 2

- Mejora dataset (más rápido), guardar mejor modelo durante el entrenamiento para cargar al final [aquí](./keras_baseline2.ipynb) -> 0.782
- Usamos todos los datos de train -> 0.844
- 3 fold cross validation, predicción ensamble [aquí](./keras_cv.ipynb): resnet50 congelada, cabeza lineal entrenada en subsets 224px, adam 1e-3, batch size 64 -> 0.811
- 5 fold cross validation, predicción ensamble, resnet50 congelada, cabeza lineal entrenada en subsets 224px, adam 1e-3, batch size 64 -> 0.827 (no hay notebook, pero lo he probado)

# Día 3

- Añadimos [data augmentation](./keras_data_augmentation.ipynb)
  - flip horizontal + flip vertical tf -> 0.7976
  - random flip keras -> 0.7873
- Añadimos [TTA](./keras_tta.ipynb) con flip horizontal y vertical -> 0.8060
- Usamos todos los datos de train -> 0.8485

