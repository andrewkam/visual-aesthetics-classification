---
layout: post
title: Balancing the Dataset
---

More images of each genre were retrieved to see if this would positively affect the classification results. As well, the same number of images for each genre was retrieved in order to create a balanced dataset:

* Country (2,000 images, 86 artists)
* R&B/Hip-hop (2,000 images, 69 artists)
* Rock (2,000 images, 60 artists)
* K-pop (2,000 images, 143 artists)

The same classifiers were run, producing the following results:

### Naive Bayes
Parameters: None
* accuracy: 0.3665
* precision_macro: 0.41612945426404774
* recall_macro: 0.3662461439273913
* f1_macro: 0.3453081120705497

### SVM
Parameters: penalty='l2', loss='squared_hinge', dual=False, C=40, max_iter=1000
* accuracy: 0.415625
* precision_macro: 0.4194762225283147
* recall_macro: 0.41595862064206857
* f1_macro: 0.41514156941272445

### Decision Tree
Parameters: criterion='entropy', max_depth=180, min_samples_split=120
* accuracy: 0.394375
* precision_macro: 0.395510021357171
* recall_macro: 0.39381256227398065
* f1_macro: 0.39351326650741103

Comparing F1 scores with the previous results, naive bayes and decision tree performance increased by a few percent. However, SVM performance only increased by 1 percent, which was disappointing, as this classifier produced the best results. While creating a larger dataset helped, it did not improve performance as much as expected!