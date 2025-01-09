# Multimodal Pathological Voice Classification

### Background
This project aims to classify pathological voice conditions using a multimodal approach, combining medical history data and voice signals. It is based on the **2023 Spring AI Cup competition** and targets five disease categories: Phonotrauma, Incomplete Glottic Closure, Vocal Palsy, Neoplasm, and Normal.

### Method
1. **Medical History Data**: Missing values were handled with median and KNN imputation. Features were processed using One Hot Encoding, MCA, and selected based on feature importance from LightGBM and Logistic Regression.

2. **Voice Signals**: Audio signals were standardized to 2-second lengths and processed using a fine-tuned Wav2Vec2 model. Dimensionality reduction was performed using PCA and feature selection, retaining 36 principal components to preserve both global and subtle features.

3. **Model Training**: Weighted sampling addressed class imbalance. Models included Logistic Regression, SVM, XGBoost, and LightGBM.  


![image](img/flow.png)

### Results
After applying the above methods, the corresponding results would have achieved a ranking of **20th out of 371** on the public leaderboard, based on the results of the past competition. LightGBM achieved the highest accuracy (0.716), while Logistic Regression demonstrated the best recall (0.603).

### Future Work
Future work includes implementing an end-to-end deep learning framework to directly optimize embeddings for classification tasks. Incorporating Cross Attention mechanisms could enhance the integration of medical history and voice features, improving the model's overall performance.

---

### How to Run
1. Run `preprocess.ipynb` to clean and preprocess medical history data.
2. Use `wav.ipynb` to process and extract features from voice signals.
3. Execute `main.ipynb` to train and evaluate the models.

