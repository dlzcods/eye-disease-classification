---
title: Eye Disease Classification
emoji: 👀
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---


# Eye Disease Classification
## Introduction
Eye health is a critical aspect of overall well-being, yet many individuals remain unaware of the conditions that can affect their vision. As the prevalence of eye diseases such as cataracts, diabetic retinopathy, and glaucoma continues to rise, early detection becomes increasingly essential. These conditions can lead to severe vision impairment or blindness if not identified and treated promptly.

**Cataracts** are characterized by clouding of the eye's lens, which can cause blurry vision, glare, and difficulty seeing at night. As people age, the risk of developing cataracts increases significantly, making timely diagnosis and intervention vital for preserving vision.

**Diabetic Retinopathy** is a complication of diabetes that affects the blood vessels in the retina. It can cause vision loss if left untreated and often has no noticeable symptoms in its early stages. Regular screening and early identification are crucial for managing this condition effectively.

**Glaucoma** refers to a group of eye conditions that damage the optic nerve, often associated with increased pressure in the eye. It can lead to irreversible vision loss if not detected early. Understanding the symptoms and risk factors for glaucoma can empower individuals to seek medical advice promptly, thereby reducing the likelihood of severe outcomes.


## About the Project
This project aims to develop a machine learning model capable of classifying eye diseases into four distinct categories: normal, cataract, diabetic retinopathy, and glaucoma. Utilizing a [dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) of over 4,000 images, the model was trained to recognize and differentiate between these conditions with high accuracy. The project employed techniques such as data augmentation to enhance the dataset and fine-tuning to optimize model performance.

## Demo
https://huggingface.co/spaces/dielz/eye-disease-classification

## Model Evaluation

### Training & Validation Accuracy
![image](https://github.com/user-attachments/assets/0b3a3039-1971-45a0-9b5b-16899a71828f)

The validation accuracy of 92.0% is quite good, showing that the model generalizes reasonably well to new data.

### Training & Validation Loss
![image](https://github.com/user-attachments/assets/b6568fee-c34b-443c-9653-f46606022160)

Validation Loss: 0.34 indicates there might still be room for improvement.

### Confusion Matrix
| Class                 | Precision | Recall   | F1-Score | Support |
|-----------------------|-----------|----------|----------|---------|
| Cataract              | 0.9278    | 0.9184   | 0.9231   | 196     |
| Diabetic Retinopathy  | 0.9756    | 0.9600   | 0.9677   | 250     |
| Glaucoma              | 0.9074    | 0.8033   | 0.8522   | 183     |
| Normal                | 0.8182    | 0.9209   | 0.8665   | 215     |
| **Accuracy**           |           |          | **0.9064** | 844     |
| **Macro Avg**         | 0.9073    | 0.9006   | 0.9024   | 844     |
| **Weighted Avg**      | 0.9096    | 0.9064   | 0.9065   | 844     |

        
![image](https://github.com/user-attachments/assets/2805a957-4825-4ec8-a909-f422ee918277)

- Cataract: Precision of 0.93 and recall of 0.92. This indicates that the model is fairly good at identifying cataract cases without too many false positives.
- Diabetic Retinopathy: Excellent precision (0.98) and recall (0.96). This suggests that the model is highly effective at detecting diabetic retinopathy.
- Glaucoma: The precision (0.91) is decent, but the recall (0.80) indicates some missed cases, which could be an area for improvement.
- Normal: A precision of 0.82 and recall of 0.92 suggests that while the model is relatively good at identifying normal cases, it may have some false positives.

The overall accuracy of 90.64% across all classes is solid, especially considering the complexity of medical imaging classification tasks.

## Model Inference
![image](https://github.com/user-attachments/assets/03374528-0daf-48c5-90cb-4e5e739c26a3)

## Conclusion
The Eye Disease Classification project demonstrates the significant potential of machine learning in improving eye disease diagnostics, achieving a strong 92% accuracy in classifying diseases like cataract, diabetic retinopathy, and glaucoma. However, there's always room for improvement to ensure the model performs even better across diverse datasets and edge cases.

To further refine the model, several advanced techniques could be employed:

- **Regularization**: Techniques like L2 regularization or dropout can be used to prevent overfitting by penalizing overly complex models, ensuring the model generalizes well on unseen data.

- **Data Augmentation**: Although already used in the project, further augmenting the dataset by applying transformations like rotations, zooms, or brightness adjustments can introduce more variability in the training data, making the model more robust to different image conditions.

- **Fine-Tuning with Transfer Learning**: Leveraging pre-trained models on larger, similar datasets and fine-tuning them for specific eye diseases could further improve accuracy, especially with smaller, domain-specific datasets.

- **Cross-Validation**: Implementing cross-validation techniques can provide a more accurate measure of the model's performance by ensuring that the model is evaluated on different subsets of the data during training.

By incorporating these techniques, the model can be further refined, enhancing its performance and adaptability in real-world medical applications. This project not only served as an important learning experience but also highlighted the vast potential machine learning holds for revolutionizing healthcare diagnostics.

