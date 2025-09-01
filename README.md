# 🧠 Parkinson’s Prediction Project (v17)

פרויקט גמר בקורס Machine Learning & AI  
הפרויקט מנתח את **UCI Parkinson’s Dataset** ובונה מודלים לחיזוי מחלת פרקינסון.  
האפליקציה נבנתה עם **Streamlit** ומאפשרת חקירה, השוואת מודלים, ניבוי והסברים.

---

## 🚀 איך מריצים?
1. התקנת חבילות:
   ```bash
   pip install -r requirements.txt
   ```
2. הרצת האפליקציה:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📊 לשוניות באפליקציה

### 1. 🔍 EDA
- התפלגות משתנים
- Boxplots לפי קבוצות
- Heatmaps
- PCA & t-SNE  

### 2. 🤖 Model Results
- השוואת ביצועים בין מודלים (RF, XGBoost, CatBoost, NN…)  
- KPIs: Accuracy, Precision, Recall, F1, ROC-AUC  

### 3. 🩺 Prediction
- ניבוי יחידני עם 🟢🟡🔴 תגיות סיכון  
- ניבוי Batch (CSV/XLSX) עם ייצוא תוצאות  

### 4. 🔄 Retrain
- אימון מודלים מחדש עם דאטה חדש  
- השוואה למודל הקיים  
- כפתור **PROMOTE** לעדכון המודל  

### 5. 🧾 Explainability
- גרפי SHAP להסבר השפעת הפיצ’רים  

### 6. 📜 Training Log
- היסטוריית אימונים עם תאריכים ומדדים

---

## 📌 הערות
- הדאטה המקורי קטן (195 דגימות).  
- המודלים מראים ביצועים טובים אך ייתכן **overfitting**.  
- לשיפור אמיתי נדרש dataset גדול יותר (קבצי קול גולמיים).  
