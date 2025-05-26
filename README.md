# Lab 4 – Visión por Computadora 2025
**Fecha:** 20 de mayo de 2025  

En este laboratorio implementará una red neuronal **U‑Net** para filtrar imágenes en escala de grises, emulando los resultados de un filtro anisotrópico de **Perona–Malik**.

---

## 1. Investigación de filtros anisotrópicos
1. Investigue qué son los filtros anisotrópicos y en qué se diferencian de filtros lineales clásicos como *Gaussian Blur*.  
2. Familiarícese con el filtro anisotrópico de **Perona–Malik**.  
3. Descargue el archivo **`Anisotropic.py`**, que ya contiene dos funciones útiles:
   * `anisodiff`  –  para imágenes en escala de grises  
   * `anisodiff3` –  para imágenes RGB  

---

## 2. Descarga del conjunto de datos BSD500
* Utilice únicamente las imágenes de prueba localizadas en:
  ```
  BSDS500/data/images/test/
  ```
* Puede clonar la réplica disponible en GitHub:  
  <https://github.com/BIDS/BSDS500>

---

## 3. Preprocesamiento  
1. Convierta cada imagen del *dataset* a **escala de grises**.  
2. Aplique el filtro anisotrópico con los parámetros sugeridos:

```python
niter  = 50
kappa  = 20
gamma  = 0.2
step   = (1., 1.)
option = 1
ploton = False
```

Esto generará, para cada imagen de entrada `Iᵢ`, una versión filtrada `Fᵢ`.

---

## 4. Generación de ventanas de entrenamiento
Para construir el *dataset* de la U‑Net:

| Símbolo | Definición |
|---------|------------|
| `xᵢ` | Ventana de tamaño **k × k** extraída de la imagen original `Iᵢ`. |
| `yᵢ` | Ventana de tamaño **k × k** en la misma posición, pero tomada de la imagen filtrada `Fᵢ`. |

* Elija `k` (p. ej. 16, 32 o 64 – potencias de 2 recomendadas).  
* Extraiga las ventanas de posiciones aleatorias **dentro** de la imagen.  
* Genere un número grande de pares `(xᵢ, yᵢ)`, por ejemplo **N = 5 × 10⁵** o más.  
* Divida el conjunto en **entrenamiento**, **validación** y **prueba**.

---

## 5. Entrenamiento de la U‑Net
* Construya una U‑Net con **3 – 4 niveles** de profundidad.  
* La red debe aceptar tensores de forma:

```
(?, k, k, 1)   # lote de ventanas de entrada
```

y producir

```
(?, k, k, 1)   # lote de ventanas filtradas
```

* Ajuste hiper‑parámetros y refine el entrenamiento hasta lograr un desempeño satisfactorio.

---

## 6. Inferencia y reconstrucción de imágenes completas
1. Deslice una ventana **k × k** sobre toda imagen de prueba (barrido).  
2. Forme un tensor de forma `(?, k, k, 1)` con todas las ventanas y páselo por la U‑Net.  
3. Reconstruya la imagen filtrada colocando cada ventana en su posición original.  
4. Donde haya solapamientos, **promedie** los valores resultantes por píxel.

---

## Organización sugerida del repositorio

```
lab04/
├── data/
│   └── bsds500/
│       └── test/
├── src/
│   ├── anisotropic.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── train_unet.py
│   └── inference.py
└── results/
    ├── originals/
    ├── filtered_gt/
    └── filtered_unet/
```

---

## Referencias
* Perona, P., & Malik, J. *Scale-space and edge detection using anisotropic diffusion*. IEEE TPAMI, 12(7), 1990.  
* BSD500 Dataset – Berkeley Segmentation Data Set and Benchmarks 500.  

---

> **Nota:** Este *README* se generó a partir del documento «Lab 4 – Visión por Computadora 2025» para proporcionar instrucciones claras y formateadas en Markdown.
