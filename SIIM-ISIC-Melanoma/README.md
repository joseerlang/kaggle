Challenge: https://www.kaggle.com/c/siim-isic-melanoma-classification/

Este es el dataset que usamos: https://www.kaggle.com/sensioai/melanoma224. Contiene las imágenes del dataset original re-escaladas a 224x224 píxeles.

# Día 1

- Intro competi
- Descarga datos
- [Exploración datos](./exploracion.ipynb)
- Sample submission -> 0.5
- [Baseline Keras](./keras_baseline.ipynb): resnet50 congelada, cabeza lineal entrenada en subset 224px, adam 1e-3, batch size 64 -> 0.764