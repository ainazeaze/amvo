Results using joint color histogram as image descriptor and a LogisticRegression for sklearn:
- Accuracy: 55.5%
- 10 classes: Faces_easy, Leopards, Motorbikes, airplanes, bonsai, car_side, chandelier, hawksbill, ketch, watch

Key observations from the confusion matrix:
- car_side performs best (20/20 correct) - easily distinguishable by color histogram
- Faces_easy also performs well (19/20)
- watch has the most confusion (12 misclassified as Motorbikes)
- bonsai struggles the most (only 1/20 correct)

----

The lower accuracy with LBP (42% vs 55.5% with color histogram) is expected for this dataset:

  1. These classes are color-discriminative:
  - car_side (19/20 with LBP, 20/20 with color): Cars have distinct colors + road/sky backgrounds
  - Faces_easy: Skin tones are very distinctive color-wise
  - Leopards: Yellow/brown spotted patterns
  - bonsai: Green vegetation is easy to distinguish by color

  2. LBP limitations for object classification:
  - Discards all color information - converts to grayscale first
  - Captures only local texture patterns (edges, spots, flat areas)
  - Uses small neighborhood (8 pixels at radius 1) - misses larger structures

  3. Look at the confusion matrix failures:
  - watch → 14 misclassified as Motorbikes (similar metallic textures/edges)
  - hawksbill (turtle) → scattered across many classes
  - chandelier → only 2/20 correct

  These objects may share similar edge/texture patterns but have very different colors.

  LBP excels at texture classification (wood vs fabric vs stone), not object recognition. For Caltech101 objects:
  - Objects have varied poses/scales → texture patterns vary
  - Background textures add noise
  - Color carries more semantic meaning for distinguishing objects

  Summary

  | Descriptor      | Best For                                                        |
  |-----------------|-----------------------------------------------------------------|
  | Color histogram | Objects with distinctive colors, scene classification           |
  | LBP             | Texture recognition, material classification, face verification |

----
Local descriptor

⏺ Results Interpretation

  Accuracy: 79.0% - A significant improvement!

  | Descriptor                  | Accuracy |
  |-----------------------------|----------|
  | LBP histogram               | 42.0%    |
  | Color histogram (global)    | 55.5%    |
  | Local descriptor (5x5 grid) | 79.0%    |

  Per-class performance:

  | Class      | Correct/Total | Accuracy |
  |------------|---------------|----------|
  | car_side   | 20/20         | 100%     |
  | Faces_easy | 19/20         | 95%      |
  | Leopards   | 18/20         | 90%      |
  | bonsai     | 17/20         | 85%      |
  | airplanes  | 16/20         | 80%      |
  | ketch      | 15/20         | 75%      |
  | Motorbikes | 14/20         | 70%      |
  | hawksbill  | 14/20         | 70%      |
  | watch      | 14/20         | 70%      |
  | chandelier | 11/20         | 55%      |


  1. Preserves spatial information: Global histogram loses WHERE colors appear. The 5x5 grid keeps location info:
    - Airplanes: blue sky at top, plane body in center
    - Faces: skin tones in center, background at edges
    - Cars: road at bottom, car body in middle
  2. More discriminative features: 12,800 dimensions (25 cells × 512 bins) vs 512 for global histogram - captures finer details
  3. Robust to partial occlusion: Each cell is independent, so one noisy region doesn't corrupt the entire descriptor
  4. Chandelier still struggles (55%): Chandeliers have highly variable shapes and backgrounds, and spatial layout varies significantly between images.

----
Bag of Visual Words (BoW) with SIFT descriptors

2. Taux de reconnaissance obtenu

Accuracy: 71.0% avec un vocabulaire de 5000 mots visuels.

3. Comparaison avec les histogrammes de couleur et LBP

| Descriptor                  | Accuracy |
|-----------------------------|----------|
| LBP histogram               | 42.0%    |
| Color histogram (global)    | 55.5%    |
| Bag of Visual Words (5000)  | 71.0%    |
| Local descriptor (5x5 grid) | 79.0%    |

Interprétation:
- BoW (71%) surpasse nettement les histogrammes de couleur (55.5%) et LBP (42%)
- Les descripteurs SIFT capturent des informations locales discriminantes (coins, contours, blobs)
- Contrairement aux histogrammes de couleur, SIFT est invariant aux changements d'illumination
- Contrairement à LBP, SIFT est partiellement invariant à l'échelle et à la rotation
- Le local descriptor (79%) reste meilleur car il préserve l'information spatiale perdue par BoW

4. Taux de reconnaissance avec différentes tailles de vocabulaire

| Vocabulary Size | Accuracy |
|-----------------|----------|
| 25              | 55.0%    |
| 50              | 56.5%    |
| 75              | 54.0%    |
| 100             | 56.0%    |
| 500             | 65.5%    |
| 1000            | 63.0%    |
| 5000            | 71.0%    |

Interprétation:
- Vocabulaires petits (25-100): ~55% - trop peu de mots visuels, perte d'information discriminante
- Vocabulaires moyens (500-1000): ~64% - meilleur compromis entre généralisation et discrimination
- Vocabulaire large (5000): 71% - meilleure performance, chaque descripteur trouve un mot visuel proche
- Le gain diminue avec la taille: 500→1000 = -2.5%, 1000→5000 = +8%
- Un vocabulaire trop petit fusionne des descripteurs différents dans le même mot visuel
- Un vocabulaire très large risque le sur-apprentissage mais ici 5000 reste optimal

----
VLAD

2. Taux de reconnaissance obtenu

Accuracy: 82.5% avec un vocabulaire de 5000 mots visuels.
Accuracy: 73% avec un vocabulaire de 5000 mots visuels avec PCA à 100 composants.

3. Comparaison VLAD vs BoW

| Descriptor                  | Accuracy |
|-----------------------------|----------|
| Bag of Visual Words (5000)  | 71.0%    |
| VLAD (5000)                 | 82.5%    |
| VLAD + PCA (100 composants) | 73.0%    |

Pourquoi VLAD (82.5%) surpasse BoW (71%):

1. **Information conservée**: BoW ne compte que les occurrences des mots visuels (histogramme).
   VLAD accumule les résidus (différences) entre chaque descripteur et son mot visuel assigné.
   → VLAD conserve l'information sur "à quel point" un descripteur diffère du centroïde.

2. **Richesse du vecteur**:
   - BoW: vecteur de taille K (5000 bins)
   - VLAD: vecteur de taille K × 128 = 640,000 dimensions
   → VLAD encode plus d'information discriminante par mot visuel.

3. **Sensibilité aux variations**: BoW perd l'information quand deux descripteurs différents
   sont assignés au même mot visuel. VLAD capture ces différences via les résidus.

4. **Normalisations**: VLAD utilise une normalisation sqrt (power normalization) qui réduit
   l'influence des mots visuels très fréquents (burstiness), puis une normalisation L2.

Pourquoi VLAD + PCA (73%) < VLAD (82.5%):

- La réduction de 640,000 → 100 dimensions perd de l'information discriminante
- Le PCA conserve la variance maximale, pas nécessairement l'information discriminante pour la classification
- 100 composants est peut-être insuffisant pour ce dataset

4. Avantages du PCA malgré la perte de précision

| Aspect | VLAD (5000) | VLAD + PCA (100) |
|--------|-------------|------------------|
| **Dimensions** | 640,000 | 100 |
| **Mémoire par image** | ~5 MB | ~0.8 KB |
| **Vitesse de classification** | Lente | ~6400x plus rapide |
| **Stockage** | Lourd | Léger |

Bénéfices clés:

1. **Scalabilité**: Avec des millions d'images, stocker des vecteurs de 640K dimensions est impraticable.
   PCA rend la recherche à grande échelle faisable.

2. **Vitesse**: Les calculs de distance (pour classification/recherche) sont O(d).
   Réduire d de 640K à 100 accélère les recherches drastiquement.

3. **Mémoire**: Un dataset de 1M images nécessite ~5TB avec VLAD brut vs ~800MB avec PCA.

4. **Généralisation**: Supprimer les dimensions à faible variance peut réduire le bruit
   et le sur-apprentissage (bien qu'ici on ait perdu trop de signal).

Le compromis 73% vs 82.5% est acceptable quand on a besoin de recherche en temps réel
sur de grandes bases de données. Pour de petits datasets où la précision prime, on évite le PCA.
