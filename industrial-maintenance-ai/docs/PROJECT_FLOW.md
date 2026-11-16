# Project workflow

1. Download AI4I 2020 CSV dataset.
2. Select machine type and sensor/process columns.
3. Split data using stratified train/test split.
4. Preprocess categorical and numerical features.
5. Train a Random Forest classifier.
6. Evaluate accuracy, precision, recall, F1 and ROC-AUC.
7. Save the trained pipeline with Joblib.
8. FastAPI loads the pipeline and exposes `/predict`.
9. React dashboard sends sensor values to the API.
10. API stores each prediction in SQLite and returns risk and maintenance guidance.
