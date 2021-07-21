# Kaggle房價預測

# 匯入所需函示庫
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import datetime

# -------------------------------------
# 執行時間紀錄

time_start = datetime.datetime.now()
print("程式執行時間: {}".format(time_start))

# -------------------------------------
# 匯入資料
test_path = r'\...\test.csv'        # path of test data
train_path = r'\...\train.csv'      # path of traning data
dt_test = pd.read_csv(test_path)
dt_train = pd.read_csv(train_path)

# -------------------------------------
# 敘述性統計
# print(dt_train.head())                        # 印出前5行
# print(dt_train.describe())                    # 敘述性統計(include="all"代表包含非計量變數)
# print(len(dt_train.columns))                  # 所有欄位名稱
# print(dt_train.shape)                         # 印出(列述、欄數)

# -------------------------------------
# 輸出報表(MAE)

table_MAE = []
table_ID = []

# -------------------------------------
# 指定y及捨去train的y
y = dt_train.SalePrice
dt_noY = dt_train.drop(['SalePrice'], axis=1)

# -------------------------------------
# (1)x篩選方式
def X_type(order):
    global dt_int, dt_test_int
    if order =='02':
        dt_int = dt_noY.select_dtypes(exclude=['object'])           #train資料，剔除非計量尺度資料
        dt_test_int = dt_test.select_dtypes(exclude=['object'])     #test資料，剔除非計量尺度資料
    
x_type_list=[
    {"ID":"02"}  #02:移除非劑量
]

# -------------------------------------
# (2)Split 拆分資料
# 在0.8、0.2下 拆成 1168和292筆資料

def Split(input_x ,input_y ,remain_size):
    global train_x, val_x, train_y, val_y
    train_x, val_x, train_y, val_y = train_test_split(input_x, input_y, train_size = remain_size, test_size = (1-remain_size), random_state= 0)

Split_list=[
    {"ID":"01","size":0.9},
    {"ID":"02","size":0.8},
    {"ID":"03","size":0.7},
    {"ID":"04","size":0.6},
    {"ID":"05","size":0.5},
]

# -------------------------------------
# (3)missing的欄位處理方式

def MissingValue(order):
    global train_x_mv, val_x_mv, x_test
    if order == "01":  
        col_missing_train = [col for col in train_x.columns if dt_int[col].isnull().any()]
        col_missing_valid = [col for col in val_x.columns if dt_int[col].isnull().any()]
        col_missing_test = [col for col in dt_test_int.columns if dt_test_int[col].isnull().any()]
        col_missing_all = set(col_missing_test) | set(col_missing_train) | set(col_missing_valid)
        train_x_mv = train_x.drop(col_missing_all, axis=1)
        val_x_mv = val_x.drop(col_missing_all, axis=1)
        x_test = dt_test_int.drop(col_missing_all, axis=1)
    elif order == '02':
        imputer = SimpleImputer(strategy = "mean")
        train_x_mv = pd.DataFrame(imputer.fit_transform(train_x))
        val_x_mv = pd.DataFrame(imputer.transform(val_x))
        x_test =  pd.DataFrame(imputer.transform(dt_test_int)) 
        train_x_mv.columns = train_x.columns
        val_x_mv.columns = val_x.columns
        x_test.columns = dt_test_int.columns
    elif order == "03":
        imputer = SimpleImputer(strategy = "median")
        train_x_mv = pd.DataFrame(imputer.fit_transform(train_x))
        val_x_mv = pd.DataFrame(imputer.transform(val_x))
        x_test =  pd.DataFrame(imputer.transform(dt_test_int)) 
        train_x_mv.columns = train_x.columns
        val_x_mv.columns = val_x.columns
        x_test.columns = dt_test_int.columns
    elif order == "04":
        imputer = SimpleImputer(strategy = "mean")     
        train_x_plus = train_x.copy()
        val_x_plus = val_x.copy()
        x_test_plus = dt_test_int.copy()
        col_missing_train = [col for col in train_x.columns if dt_int[col].isnull().any()]
        col_missing_valid = [col for col in val_x.columns if dt_int[col].isnull().any()]
        col_missing_test = [col for col in dt_test_int.columns if dt_test_int[col].isnull().any()]
        col_missing_all = set(col_missing_test) | set(col_missing_train) | set(col_missing_valid) 
        for col in col_missing_all:
            train_x_plus[col+'_was_missing'] = train_x_plus[col].isnull()
            val_x_plus[col+'_was_missing'] = val_x_plus[col].isnull()
            x_test_plus[col+'_was_missing'] = x_test_plus[col].isnull()
        train_x_mv = pd.DataFrame(imputer.fit_transform(train_x_plus))
        val_x_mv = pd.DataFrame(imputer.transform(val_x_plus))      
        x_test =  pd.DataFrame(imputer.transform(x_test_plus))       
        train_x_mv.columns = train_x_plus.columns
        val_x_mv.columns = val_x_plus.columns
        x_test.columns = x_test_plus.columns     
    elif order == "05":       
        imputer = SimpleImputer(strategy = "median")     
        train_x_plus = train_x.copy()
        val_x_plus = val_x.copy()
        x_test_plus = dt_test_int.copy()
        col_missing_train = [col for col in train_x.columns if dt_int[col].isnull().any()]
        col_missing_valid = [col for col in val_x.columns if dt_int[col].isnull().any()]
        col_missing_test = [col for col in dt_test_int.columns if dt_test_int[col].isnull().any()]
        col_missing_all = set(col_missing_test) | set(col_missing_train) | set(col_missing_valid) 
        for col in col_missing_all:
            train_x_plus[col+'_was_missing'] = train_x_plus[col].isnull()
            val_x_plus[col+'_was_missing'] = val_x_plus[col].isnull()
            x_test_plus[col+'_was_missing'] = x_test_plus[col].isnull()
        train_x_mv = pd.DataFrame(imputer.fit_transform(train_x_plus))
        val_x_mv = pd.DataFrame(imputer.transform(val_x_plus))      
        x_test =  pd.DataFrame(imputer.transform(x_test_plus))       
        train_x_mv.columns = train_x_plus.columns
        val_x_mv.columns = val_x_plus.columns
        x_test.columns = x_test_plus.columns   
        
MV_list=[
    {"ID":"01"},         #01:drop train和test丟失的資料
    {"ID":"02"},         #02:Imputate"Mean"
    {"ID":"03"},         #02:Imputate"Median"
    {"ID":"04"},         #02:Imputate"Mean" + Dummy"missing"
    {"ID":"05"},         #02:Imputate"Median" + Dummy"missing"
]

# -------------------------------------
# (4)模型設定

def RF(n):
    model = RandomForestRegressor(n_estimators= n, random_state= 0)
    return model

model_list=[
    {'ID':'01','n_estimators':10},
    {'ID':'02','n_estimators':30},
    {'ID':'03','n_estimators':50},
    {'ID':'04','n_estimators':70},
    {'ID':'05','n_estimators':90},
    {'ID':'06','n_estimators':110},
    {'ID':'07','n_estimators':130},
    {'ID':'08','n_estimators':150},
    {'ID':'09','n_estimators':170},
    {'ID':'10','n_estimators':190},
]

# -------------------------------------
# MAE設定

def cal_MAE():
    score_MAE = mean_absolute_error(val_y, model_pred)
    return score_MAE
    
# # -------------------------------------
# 印ID

def ID_merge(a, b, c, d):
    ID = a+b+c+d
    return ID

# # -------------------------------------
# 批次模型試跑

def Run_all():
    global model_pred
    for i in range(len(x_type_list)):
        X_type(x_type_list[i]['ID'])
        for j in range(len(Split_list)):
            Split(dt_int, y, Split_list[j]['size'])
            for k in range(len(MV_list)):
                MissingValue(MV_list[k]['ID'])
                for l in range(len(model_list)):
                    model = RF(model_list[l]['n_estimators'])
                    ID = ID_merge(x_type_list[i]['ID'], Split_list[j]['ID'], MV_list[k]['ID'], model_list[l]['ID'])
                    model.fit(train_x_mv, train_y)
                    model_pred = model.predict(val_x_mv)
                    table_MAE.append(cal_MAE())
                    table_ID.append(ID)
                    print("ID: {} / Mean Absolute Error(MAE):{}".format(ID, cal_MAE()))
    # # -------------------------------------
    # 測試結果存檔
    test_list={"ID":table_ID, "MAE":table_MAE}
    df = pd.DataFrame(test_list)
    with pd.ExcelWriter('C://Users/Rex/Desktop/Rex/Python-training/kaggle/house_price_predict/test/Run_all_{}.xlsx'.format(time_start.strftime('%Y-%m-%d-%H-%M'))) as writer:  
        df.to_excel(writer, sheet_name='Sheet1')

def Run_one():
    global model_pred
    for item in Run_one_list:
        ID_a = item['ID'][0:2]
        ID_b = item['ID'][2:4]
        ID_c = item['ID'][4:6]
        ID_d = item['ID'][6:8]
        X_type(ID_a)
        Split(dt_int, y, call_split(ID_b))
        MissingValue(ID_c)
        model = RF(call_model(ID_d))
        ID = item['ID']
        model.fit(train_x_mv, train_y)
        model_pred = model.predict(val_x_mv)
        table_MAE.append(cal_MAE())
        table_ID.append(ID)
        print("ID: {} / Mean Absolute Error(MAE):{}".format(ID, cal_MAE()))
    # # -------------------------------------
    # 測試結果存檔
    test_list={"ID":table_ID, "MAE":table_MAE}
    df = pd.DataFrame(test_list)
    with pd.ExcelWriter('C://Users/Rex/Desktop/Rex/Python-training/kaggle/house_price_predict/test/Run_one_{}.xlsx'.format(time_start.strftime('%Y-%m-%d-%H-%M'))) as writer:  
        df.to_excel(writer, sheet_name='Sheet1')

Run_one_list=[
    {"ID":"02010410"}
]

def Predict_Genernate():
    global model_pred
    for item in Predict_generate_list:
        ID_a = item['ID'][0:2]
        ID_b = item['ID'][2:4]
        ID_c = item['ID'][4:6]
        ID_d = item['ID'][6:8]
        X_type(ID_a)
        Split(dt_int, y, call_split(ID_b))
        MissingValue(ID_c)
        model = RF(call_model(ID_d))
        ID = item['ID']
        model.fit(train_x_mv, train_y)
        model_pred = model.predict(val_x_mv)
        table_MAE.append(cal_MAE())
        table_ID.append(ID)
        print("ID: {} / Mean Absolute Error(MAE):{}".format(ID, cal_MAE()))
        model_pred_test = model.predict(x_test)        
        output = pd.DataFrame({'Id': x_test['Id'], 'SalePrice': model_pred_test})
        output['Id'] = output['Id'].astype(int)
        output_path = 'C:\\Users\Rex\Desktop\Rex\Python-training\kaggle\house_price_predict\submission\submission_{}_{}.csv'.format(time_start.strftime('%Y-%m-%d-%H-%M'), ID)
        output.to_csv(output_path, index= False)
        print("ID: {} / 已成功輸出預測結果！".format(ID))

Predict_generate_list=[
    {"ID":"02010405"},
]

# # -------------------------------------
# order對應參數

def call_model(order):
    for item in model_list:
        if item['ID'] == order:
            return item['n_estimators']
        else:
            continue

def call_split(order):
    for item in Split_list:
        if item['ID'] == order:
            return item['size']
        else:
            continue            


# # -------------------------------------
# 主程序

# Run_all()
# Run_one()
# Predict_Genernate()

# # -------------------------------------

time_end = datetime.datetime.now()
time_pass = time_end - time_start
print("程式結束時間: {}".format(time_end))
print("總共花費: {}".format(time_pass))
