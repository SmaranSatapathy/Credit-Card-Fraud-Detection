# 💳 Fraud Detection System using Neural Networks
> **Detecting the undetectable** – Building an intelligent credit card fraud detection system leveraging deep learning to secure digital transactions.

---

## 🚀 Project Overview

In the digital economy, fraudulent transactions pose a critical threat to individuals and financial institutions. Traditional rule-based detection systems often fail to detect complex fraud patterns. This project develops a robust **Fraud Detection System using Neural Networks**, capable of identifying fraudulent transactions with high accuracy.

---

## 🎯 Objectives

✅ Study real-world transaction data and understand fraud behaviour  
✅ Design a layered **Artificial Neural Network** to classify transactions  
✅ Handle **class imbalance** using **SMOTE** and **Random UnderSampling**  
✅ Evaluate using precision, recall, F1-score, ROC-AUC, and accuracy

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries & Frameworks:** TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imblearn  
- **Environment:** Google Colab (GPU accelerated)

---

## 📂 Dataset

- **Source:** European cardholder transactions (Sept 2013)  
- **Size:** ~285K transactions  
- **Fraud Ratio:** ~0.172%  
- **Features:** 28 anonymised PCA components + Time + Amount + Class (fraud/not fraud)

---

## 🧠 Model Architecture

- Input layer with **30 features**  
- Hidden Layer 1: 64 neurons (ReLU/Tanh)  
- Hidden Layer 2: 32 neurons (ReLU/Tanh)  
- **Dropout:** 0.5 after each hidden layer  
- Output Layer: Sigmoid activation for binary classification  
- **Loss Function:** Binary Crossentropy  
- **Optimizers experimented:** SGD, Adam, RMSProp

---

## ⚙️ Preprocessing Pipeline

1. **EDA**: Statistical summary, class imbalance check  
2. **Normalization**: StandardScaler to standardize features  
3. **Hybrid Sampling**:
   - **SMOTE** to oversample fraud cases
   - **RandomUnderSampler** to reduce majority class  
4. Train-Test split and model training with callbacks

---

## 📈 Results

- Achieved balanced fraud detection performance across optimizers
- ROC-AUC, precision, recall, and F1-score evaluated for each configuration
- Hybrid sampling improved detection rate significantly over raw data

---

---

## 💡 Future Scope

✨ Real-time deployment using **Apache Kafka / Flink**  
✨ Integration with banking systems or cloud platforms (AWS/GCP)  
✨ Advanced architectures – **LSTM, RNNs, Autoencoders** for temporal pattern analysis  
✨ Hybrid models combining ANN with **rule-based systems** for comprehensive detection

---

## 👨‍💻 Authors

- Smaran Satapathy
- Ansujit Dalei  
- Subhasish Nahak  

> Guided by **Jagseer Singh Sir**, Department of CSE (AI & ML), Siksha 'O' Anusandhan University

---

## 📚 References

This project is inspired by state-of-the-art research in fraud detection using machine learning and deep learning approaches. For detailed references, check the [project report](./Fraud_Card_Detection_Project_Report.pdf).

---

## 🔗 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

> **“Fraudsters innovate, so should detection.”** – This project demonstrates how deep learning empowers automated, scalable, and intelligent fraud detection for tomorrow’s secure digital economy.
