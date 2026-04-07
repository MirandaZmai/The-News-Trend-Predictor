# The News Trend Predictor 📈
**An AI-driven R&D project for forecasting the popularity trajectory of news items.**

## 📖 Overview
**The News Trend Predictor** is a web-based analytical tool designed for journalists, social media marketers, and content creators. It predicts the future relevancy of news topics based on their recent Google Search trajectories. By forecasting the **Google Trend Score (0-100)**, the app empowers users to optimize their **SEO strategies** and align content creation with algorithmic discovery patterns.

---
## 🚀 Key Features
* **Trend Forecasting:** Predicts the popularity of specific news text strings using historical search data.
* **Multi-Platform Data Fusion:** Integrates data from Google Trends and YouTube to capture cross-platform interest.
* **Time-Series Analysis:** Categorizes news into short, medium, and long-term items for tailored prediction methodologies.
* **SEO Optimization:** Provides actionable insights for content creators to time their publishing for maximum reach.
  
---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Acquisition:** * `PyTrends` (Google Trends API Scraper)
    * `YouTube Data API` (Engagement metrics)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Regression & Time-series forecasting)
* **Deployment:** Web App Framework (Streamlit/Flask)

---

## 📊 Methodology & Research
Our R&D process focused on four primary research questions:
1.  **Effectiveness:** Validating prediction algorithms for individual news popularity.
2.  **Data Quality:** Assessing how source reliability impacts forecast accuracy.
3.  **Categorization:** Understanding the relationship between news types and trend longevity.
4.  **Model Selection:** Benchmarking ML algorithms to find the optimal fit for normalized trend data.

### **Datasets Used**
* **Google Search Trend Data:** Normalized popularity scores (0-100).
* **Custom Calendar Dataset:** Internally generated to account for temporal seasonality.
* **YouTube Engagement Data:** Keyword-related video performance during specific timeframes.

---

## 📈 Key Findings & Insights
* **Overfitting Management:** Identified that high-correlation features in small datasets can lead to poor generalization; implemented rigorous feature selection to mitigate this.
* **Dynamic Relevancy:** Successfully demonstrated that combining YouTube engagement with Google Trends provides a more robust signal for "viral" potential than search data alone.
* **Model Performance:** Determined that the choice of algorithm must vary based on the "lifespan" (short-term vs. long-term) of the news category.

---

### **💡 Portfolio Note**
This project demonstrates a full-cycle Data Engineering and Machine Learning workflow—from multi-source data scraping and cleaning to predictive modeling and deployment. It bridges the gap between **Information Science** and **Content Marketing Strategy**.
