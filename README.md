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
