---
layout: post
title: First Trial
---

As a trial to see if my genre classification idea would even work, photos from the following four genres were retrieved:
* Country (1,072 images, 40 artists)
* R&B/Hip-hop (1,305 images, 37 artists)
* Rock (1,394 images, 36 artists)
* K-pop (1,138 images, 74 artists)

Artist lists for country, R&B/hip-hop, and rock were pulled form Billboard charts Country Albums, R&B Albums, Rock Albums. K-pop artists were retrieved from the Gaon Digital Chart. Images were resized to 300x300, and processed using the clothing image classification model found [here](https://www.deepdetect.com/applications/model/).

After image detection, the following classifiers were run:

### Naive Bayes
Parameters: None
* accuracy: 0.33469546176416365
* precision_macro: 0.3761609587593987
* recall_macro: 0.34189628734972566
* f1_macro: 0.3131771098207531

### SVM
Parameters: penalty='l2', loss='squared_hinge', dual=False, C=1, max_iter=1000
* accuracy: 0.41536640557843557
* precision_macro: 0.41448550024974046
* recall_macro: 0.4059822336174907
* f1_macro: 0.4053436576137777

### Decision Tree
Parameters: criterion='gini', max_depth=18, min_samples_split=50
* accuracy: 0.3815432710722263
* precision_macro: 0.4315133134688377
* recall_macro: 0.36902721426276114
* f1_macro: 0.34820470527426756