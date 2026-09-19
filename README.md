\# 🌾 AgriYield AI



AgriYield AI is a machine-learning-based multi-crop yield prediction system developed as part of an AIML internship project.



\## Project Title



\*\*Strategic Planning and Development of a Machine Learning-Based Multi-Crop Yield Prediction System for Agriculture\*\*



\## Features



\- Multi-crop yield prediction

\- Crop-specific machine learning models

\- Historical agricultural analysis

\- Temporal model validation

\- Interactive Streamlit web interface

\- State and season-based prediction

\- Model performance visualization



\## Dataset



The project uses historical agricultural data containing:



\- 19,689 records

\- 55 crops

\- 30 states

\- 6 seasons

\- Years 1997–2020



\## Machine Learning Models



The following algorithms were evaluated:



\- Linear Regression

\- Random Forest Regression

\- Gradient Boosting Regression



Models were evaluated using temporal validation:



\- Training: 1997–2017

\- Testing: 2018–2020



Only crop models with positive temporal R² were included in the final application.



\## Supported Validated Crops



\- Rice

\- Moong (Green Gram)

\- Urad

\- Groundnut

\- Potato

\- Sugarcane

\- Wheat

\- Rapeseed \& Mustard



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Streamlit

\- Joblib



\## Run the Application



Install dependencies and run:



```bash

streamlit run app.py

