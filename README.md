# 🛒 E-Commerce Customer Purchase Prediction

Predict customer purchase intent in an e-commerce environment using session and behavioral data. 🎯

---
## 📌 Project Overview

This project focuses on **predicting whether a website visitor will make a purchase** during an online shopping session.

By identifying visitors with high purchase intent, businesses can optimize **targeting, personalization, and retargeting strategies**, ultimately boosting **revenue and conversion rates**. 💰

**Target:**  
- 🟢 <span style="color:green"><b>Buy</b></span>  
- 🔴 <span style="color:red"><b>Not Buy</b></span>


**Business Impact:**  
- 🎯 Improved marketing targeting  
- ✨ Personalized user experience  
- 📈 Higher conversion rates and sales  

---
## 📂 Dataset

We use the **Online Shoppers Purchasing Intention Dataset**, which contains detailed information about visitor sessions.  

**Features include:**  
- 📝 Pages visited  
- ⏱️ Session duration  
- 💳 Cart value  
- 📱 Device type  
- 🌐 Traffic source  
- 🛍️ Past purchases  

**Sources:**  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
- [Kaggle Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)  

---
## 🔑 Features

- `Administrative`, `Informational`, `ProductRelated` pages viewed  
- `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`  
- `BounceRates`, `ExitRates`, `PageValues`  
- `SpecialDay` (seasonal promotions) 🎉  
- `Month` of the visit 📆  
- `OperatingSystems`, `Browser`, `Region`, `TrafficType` 🖥️  
- `VisitorType` (New/Returning) 👤  
- `Weekend` indicator 🛌  

---
## ⚙️ Methodology

1. **Data Preprocessing** 🧹  
   - Handle missing values  
   - Encode categorical features  
   - Normalize numerical features  

2. **Exploratory Data Analysis (EDA)** 🔍  
   - Analyze session patterns  
   - Identify trends and correlations with purchase behavior  

3. **Modeling** 🤖  
   - Classification models (e.g., Logistic Regression, Random Forest, XGBoost)  
   - Model evaluation with accuracy, precision, recall, F1-score  

4. **Deployment (Optional)** 🚀  
   - Predict purchase intent in real-time for live sessions  
   - Integrate predictions with personalization and marketing tools  

---
## 🎯 Goals

- Identify sessions with high likelihood of purchase 🛍️  
- Optimize marketing spend and retargeting campaigns 💸  
- Increase conversion rate and customer satisfaction 😃  

---
## 📜 License

This project uses publicly available datasets.

---