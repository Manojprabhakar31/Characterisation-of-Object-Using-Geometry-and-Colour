# Characterisation of Object Using Geometry and Colour

### Overview

This project focuses on **object characterization and classification** using a combination of **geometric shape descriptors** and **colour features** extracted from images.
The pipeline is designed to be **scale, rotation, and translation-invariant**, making it suitable for applications such as industrial inspection, robotics, and automated sorting.

## Feature Extraction Pipeline

<div align="center">

  <img src="https://github.com/user-attachments/assets/66135706-3318-446d-8d3f-b76f873dea53"
       width="900" style="height:auto;" />

  <p><i>
    Figure 1: Feature extraction pipeline for the IISc cup object, showing the sequence of image processing and geometric feature computation.
  </i></p>

  <br>

  <img src="https://github.com/user-attachments/assets/8c091ebf-e094-4957-b2ff-3c43460bc11f"
       width="900" style="height:auto;" />

  <p><i>
    Figure 2: Feature extraction pipeline for the scissor object, illustrating the same processing stages applied to a different object class.
  </i></p>

</div>


## Final Feature Vector (22 Features)

The object is represented using a **22-dimensional feature vector**, combining geometry and colour information:

| Feature Category             | Description              | Count  |
| ---------------------------- | ------------------------ | ------ |
| PCA Eigenvalues              | Principal shape spread   | 2      |
| Eigenvalue Ratio             | Shape elongation         | 1      |
| Fourier Descriptors          | Boundary shape details   | 10     |
| Global Shape + Aspect Ratio  | Overall object geometry  | 2      |
| Local Shapes + Aspect Ratios | Sub-region shape details | 4      |
| Colour Features              | Mean L, a, b (LAB space) | 3      |
| **Total**                    |                          | **22** |


## Classification

Two supervised classifiers are evaluated:

* **Support Vector Machine (SVM)**
* **K-Nearest Neighbour (KNN)**

<div align="center">

<img src="https://github.com/user-attachments/assets/6c25ae3e-6542-48c1-a179-13d4216650a4"
  width="900" style="height:auto;" />

<p><i>Figure 3: PCA feature space visualization using selected shape features.</i></p>

<br>

<img src="https://github.com/user-attachments/assets/a1e08d85-3acf-49ab-8e15-a857ceaa9984"
  width="900" style="height:auto;" />

<p><i>Figure 4: Classification performance comparison using different feature combinations.</i></p>

</div>

## Results Summary

* Shape-only features provide **moderate classification accuracy**
* Adding **global and local shape descriptors** significantly improves performance
* **Best accuracy is achieved when colour (LAB) features are combined with shape features**
* PCA visualization shows improved class separability with richer feature sets

## Limitations

* Designed for **one object per image**
* Requires a **uniform background**
* Operates only on **colour images**
* Performance depends on dataset diversity
* PCA space changes with training data distribution

## Conclusion

This work presents a **robust and interpretable object characterization pipeline** using:

* PCA-based geometric descriptors
* Fourier boundary analysis
* Global and local shape modeling
* LAB colour statistics

The fusion of **geometry and colour features** significantly improves classification accuracy while maintaining invariance to scale, rotation, and position.

## Future Scope

* Improve handling of concave and complex shapes
* Extend to real-time webcam-based classification
* Reduce dependence on colour features
* Generalize to unseen object categories
* Explore hybrid deep-learning + handcrafted features
