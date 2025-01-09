# Multimodal Pathological Voice Classification

### Background
This project classifies pathological voice conditions by combining medical history data and voice signals. It is based on the **2023 Spring AI Cup competition**, targeting five disease categories: Phonotrauma, Incomplete Glottic Closure, Vocal Palsy, Neoplasm, and Normal.

### Method
1. **Medical History Data**: Missing values were imputed using median and KNN methods. Features were encoded with One Hot Encoding, MCA, and selected using LightGBM and Logistic Regression.

2. **Voice Signals**: Audio signals were standardized to 2-second lengths and processed using a fine-tuned Wav2Vec2 model. Dimensionality reduction was performed using PCA and feature selection, retaining 36 principal components to preserve both global and subtle features.

3. **Model Training**: Weighted sampling addressed class imbalance. Models included Logistic Regression, SVM, XGBoost, and LightGBM.  


![image](img/flow.png)

### Results
The methods would rank **20th out of 371** on the past competition's public leaderboard. LightGBM achieved the highest accuracy (0.716), while Logistic Regression had the best recall (0.603).

### Future Work
Future work involves designing an end-to-end deep learning model to optimize embeddings directly, incorporating Cross Attention to enhance integration between modalities.

---

### How to Run
1. Run `preprocess.ipynb` to clean and preprocess medical history data.
2. Use `wav.ipynb` to process and extract features from voice signals.
3. Execute `main.ipynb` to train and evaluate the models.

