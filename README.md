# CLIP vs. the Classics: Augmenting Traditional Classifiers with Modern Feature Representations
In our project, we will first compare zero-shot CLIP classification to models considered state-of-the-art over a decade
ago. Second, we will see if using CLIP features with these models can outperform ResNet50 on medical image classification
tasks. Our work will study two datasets: HAM10000 for skin lesion classification and the NIH Chest X-ray dataset.
The state-of-the-art approaches to these classification tasks in the 2000s used random forests and SVMs combined with
useful, feature selection techniques. We will re-implement these to conduct our tests. While these methods alone cannot
stand up to modern deep learning techniques, combining older models, including SVMs, decision trees, and clustering with
CLIP feature extraction, will help them compete with newer techniques.

We chose datasets that would have been relevant when the older papers we were examining were written. X-ray seg-
mentation was one of the early applications of computer vision, and skin lesion classification became relevant later. The
Chest X-ray dataset labels images with classes corresponding to 14 diseases and a “No Findings” label. The diseases include
Edema, Fibrosis, Pneumonia, and Hernia. The HAM10000 dataset contains dermatoscopic images of skin lesions with similar 
labels for seven diseases, including Actinic keratoses, basil cell carcinoma, and melanoma. We hope each domain will
provide helpful information about the efficacy of the aforementioned models on medical data.

Our results showed that using CLIP and MedCLIP features boosted performance, enabling the older models to achieve
almost as high accuracy as ResNet, as seen in our report. We believe that this technique offers a valuable alternative to deep
neural networks when runtime efficiency is prioritized over accuracy.

See the report in Report.pdf
