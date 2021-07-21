# Kaggle_model
# 為一自動化處理之批次生成RandomForest模型及其MAE
# 可藉由本程式找出MAE最佳解（Kaggle: House Prices - Advanced Regression Techniques - (前60%)）

# 內部調整參數
# 1.移除類別變數
# 2.調整資料Split比例
# 3.MissingValue處理(drop、mean、medium、dummy variable)
# 4.RandomForestRegressor之n_estimators數量
