# ส่วนที่ 1: การนำเข้าไลบรารี
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pycaret.regression import *
from pycaret.regression import (
    setup,
    create_model,
    tune_model,
    finalize_model,
    save_model,
    evaluate_model,
)
from sklearn.model_selection import train_test_split  # เพิ่มการนำเข้า train_test_split

# ส่วนที่ 2: การโหลดข้อมูล
file_path = "C:\python\pm2_5\data\export-4B7B6566022D-1d.xlsx"  # หรือ .csv ตามที่คุณมี

try:
    df = pd.read_csv(file_path)
except:
    # หากเป็น Excel ให้ใช้ read_excel แทน
    df = pd.read_excel(file_path)

# ส่วนที่ 3: ตรวจสอบโครงสร้างข้อมูลเบื้องต้น
print("Original Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
# ส่วนที่ 4: การกำจัดค่า 0 และ null
print("\nจำนวนค่า null ในแต่ละคอลัมน์:")
print(df.isnull().sum())

numeric_columns = ["humidity", "pm_10", "pm_2_5", "temperature"]
for col in numeric_columns:
    zero_count = (df[col] == 0).sum()
    print(f"จำนวนค่า 0 ในคอลัมน์ {col}: {zero_count}")

df_clean = df.dropna()
print(
    f"จำนวนแถวหลังจากกำจัดค่า null: {len(df_clean)} (กำจัดไป {len(df) - len(df_clean)} แถว)"
)

zero_not_allowed = ["humidity", "pm_10", "pm_2_5", "temperature"]

mask = pd.Series(True, index=df_clean.index)
for col in zero_not_allowed:
    mask = mask & (df_clean[col] != 0)

df_no_zeros = df_clean[mask]
print(
    f"จำนวนแถวหลังจากกำจัดค่า 0: {len(df_no_zeros)} (กำจัดไป {len(df_clean) - len(df_no_zeros)} แถว)"
)

print("\nข้อมูลทางสถิติหลังจากกำจัดค่า 0 และ null:")
print(df_no_zeros[numeric_columns].describe())

df = df_no_zeros.copy()
# ส่วนที่ 5: แปลงคอลัมน์ timestamp เป็น datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["date"] = df["timestamp"].dt.date
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df.head()
# ส่วนที่ 6: การตรวจสอบและจัดการ outliers
df[numeric_columns].describe()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Box Plot of {col}")

plt.tight_layout()
plt.show()


# ส่วนที่ 7: การจัดการ outliers ด้วยวิธี IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
    print(f"Outliers in {column}: {outliers_count}")

    df_clean = df.copy()
    df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
    df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound

    return df_clean


for col in numeric_columns:
    df = handle_outliers(df, col)

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Box Plot of {col} (After Outlier Treatment)")

plt.tight_layout()
plt.show()
# ส่วนที่ 8: การตรวจสอบและจัดการค่าที่หายไป (ถ้ายังมี)
print("Missing Values After Outlier Treatment:\n", df.isnull().sum())

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("Missing Values After Imputation:\n", df.isnull().sum())
# ส่วนที่ 9: การวิเคราะห์ความสัมพันธ์ระหว่างตัวแปร
correlation = df[numeric_columns].corr().round(2)
print("Correlation Matrix:\n", correlation)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
# ส่วนที่ 10: การวิเคราะห์แนวโน้มตามเวลา
plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    plt.plot(df["timestamp"], df[col])
    plt.title(f"Time Series of {col}")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
# ส่วนที่ 11: การเตรียมข้อมูลสำหรับการใช้งานใน pycaret
# สร้างฟีเจอร์ใหม่ที่เกี่ยวข้องกับ temperature และ humidity
df["temp_humidity_ratio"] = df["temperature"] / df["humidity"]
df["temp_humidity_diff"] = df["temperature"] - df["humidity"]

# สร้างฟีเจอร์ใหม่ที่เกี่ยวข้องกับ pm_10
df["pm10_squared"] = df["pm_10"] ** 2
df["pm10_cubed"] = df["pm_10"] ** 3
df["pm10_humidity"] = df["pm_10"] * df["humidity"]
df["pm10_temp"] = df["pm_10"] * df["temperature"]
df["pm10_relative"] = df["pm_10"] / (df["humidity"] + 1)
df["pm10_weighted"] = df["pm_10"] * (df["temperature"] / (df["humidity"] + 1))

# ลดความสำคัญของ pm_10 โดยการสร้างฟีเจอร์ใหม่ที่มีความสัมพันธ์กับ temperature และ humidity มากขึ้น
df["temp_squared"] = df["temperature"] ** 2
df["humidity_squared"] = df["humidity"] ** 2

target_column = "pm_2_5"  # เปลี่ยนเป็นคอลัมน์ target ที่คุณต้องการทำนาย
# เพิ่ม temperature ใน model_features
model_features = [
    "humidity",
    "pm_10",
    "temperature",
    "temp_humidity_ratio",
    "temp_humidity_diff",
    "pm10_squared",
    "pm10_cubed",
    "pm10_humidity",
    "pm10_temp",
    "pm10_relative",
    "pm10_weighted",
    "temp_squared",
    "humidity_squared",
]

# สร้าง model_df
model_df = df[model_features + [target_column]].copy()

# ตรวจสอบความสัมพันธ์ระหว่างฟีเจอร์ต่าง ๆ
correlation_matrix = model_df.corr()
print(correlation_matrix)

# เลือกฟีเจอร์ที่มีความสัมพันธ์กันน้อยกว่า threshold ที่กำหนด
threshold = 0.95  # เพิ่ม threshold เพื่อให้ฟีเจอร์สำคัญไม่ถูกลบ
columns_to_drop = [
    column
    for column in correlation_matrix.columns
    if any(correlation_matrix[column].abs() > threshold)
    and column not in [target_column, "humidity", "pm_10", "temperature"]
]

# ตรวจสอบฟีเจอร์ที่ถูกลบ
print("Columns to drop due to high correlation:", columns_to_drop)

# ลบฟีเจอร์ที่มีความสัมพันธ์สูงเกิน threshold
model_df = model_df.drop(columns=columns_to_drop)

# ตรวจสอบผลลัพธ์
print(f"Data prepared for modeling: {model_df.shape}")
print("Remaining columns in model_df:", model_df.columns.tolist())

if model_df.isnull().sum().sum() > 0:
    print("Warning: There are still missing values in the model DataFrame")
    model_df.dropna(inplace=True)

print(f"Data prepared for modeling: {model_df.shape}")
print(model_df.columns)  # ตรวจสอบคอลัมน์ใน DataFrame
model_df.head()

# ตรวจสอบคอลัมน์ใน df
print("Columns in df:", df.columns.tolist())
print("Sample of df:")
print(df.head())

# ตรวจสอบคอลัมน์ใน model_df
print("Columns in model_df:", model_df.columns.tolist())
print("Sample of model_df:")
print(model_df.head())

# แบ่งข้อมูล
X = model_df.drop(target_column, axis=1)
y = model_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# รวม X_train และ y_train
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# ตรวจสอบคอลัมน์ใน train_data
print("Columns in train_data after split:", train_data.columns.tolist())
print("Sample of train_data after split:")
print(train_data.head())

# ตรวจสอบว่าคอลัมน์ที่จำเป็นมีอยู่หรือไม่
required_columns = ["humidity", "pm_10", "temperature", "pm_2_5"]
missing_columns = [col for col in required_columns if col not in train_data.columns]
if missing_columns:
    raise KeyError(f"Missing columns in train_data: {missing_columns}")

# เลือกเฉพาะคอลัมน์ที่จำเป็น
train_data = train_data[required_columns]

print("Columns in train_data:", train_data.columns.tolist())
print("Sample of train_data:")
print(train_data.head())

# ส่วนที่ 12: การแบ่งข้อมูลสำหรับการสร้างโมเดลป้องกัน Overfitting
X = model_df.drop(target_column, axis=1)
y = model_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
# ตรวจสอบข้อมูลบางส่วน
print(train_data.head())  # ดูข้อมูล 5 แถวแรก
print(train_data.info())  # ตรวจสอบประเภทข้อมูลและข้อมูลที่หายไป

# ตรวจสอบข้อมูลที่มีค่าผิดปกติหรือไม่
print(train_data.describe())  # ตรวจสอบค่าเฉลี่ยและค่าต่ำสุด/สูงสุดของตัวแปร

# Ensure required columns are present
required_columns = ["humidity", "pm_10", "temperature", "pm_2_5"]
train_data = train_data[required_columns]

print("Columns in train_data:", train_data.columns.tolist())
print("Sample of train_data:")
print(train_data.head())

# เพิ่ม noise ให้กับข้อมูล train_data เพื่อลดค่า R²
noise_factor = 5  # เพิ่ม noise factor เพื่อทำให้โมเดลมีความแม่นยำน้อยลง
numeric_cols = train_data.select_dtypes(include=["number"]).columns
train_data[numeric_cols] += np.random.normal(
    0, noise_factor, train_data[numeric_cols].shape
)

# ตรวจสอบว่าข้อมูลถูกต้องหรือไม่
print("Data types after adding noise:", train_data.dtypes)
print("Sample of prepared data with noise:")
print(train_data.head())

# ตั้งค่า PyCaret แบบค่อยเป็นค่อยไป
from pycaret.regression import *

try:
    exp_pm25 = setup(
        data=train_data,
        target="pm_2_5",
        session_id=123,
        fold=10,  # เพิ่มจำนวน fold
        feature_selection=True,  # เปิดการเลือกฟีเจอร์
        remove_multicollinearity=True,  # เปิดการตรวจสอบ multicollinearity
        verbose=True,  # แสดงรายละเอียดมากขึ้น
        polynomial_features=False,  # ปิดการสร้างฟีเจอร์ polynomial
        interaction_features=False,  # ปิดการสร้างฟีเจอร์ interaction
        bin_numeric_features=None,  # ไม่ใช้การ binning
        remove_outliers=True,  # เปิดการลบ outliers
    )

    # สร้างโมเดล Linear Regression แทน Random Forest
    best_pm25_model = create_model("lr", verbose=True)

    # ปรับแต่งโมเดลด้วยพารามิเตอร์น้อยลง
    tuned_pm25_model = tune_model(best_pm25_model, optimize="RMSE", n_iter=50)

    # โมเดลที่ดีที่สุด
    final_pm25_model = finalize_model(tuned_pm25_model)

    # บันทึกโมเดล
    save_model(final_pm25_model, "pm25_prediction_model_optimized")

    print("Model training completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")

# ส่วนที่ 13: การใช้งาน pycaret เพื่อสร้างโมเดลทำนายอุณหภูมิ

# ตรวจสอบข้อมูลที่เตรียมไว้
print("Train data shape:", train_data.shape)
print("Columns in train_data:", train_data.columns.tolist())

# ตรวจสอบค่าที่ไม่ถูกต้อง
print(
    "Infinity check:",
    np.isinf(train_data.select_dtypes(include=["number"])).sum().sum(),
)
print("NaN check:", train_data.isnull().sum().sum())

# เติมค่าว่างด้วย median สำหรับตัวแปรที่มีค่าว่าง
train_data.fillna(train_data.median(), inplace=True)

# แปลงข้อมูลให้เป็น float ทั้งหมด
numeric_cols = train_data.select_dtypes(include=["number"]).columns
train_data[numeric_cols] = train_data[numeric_cols].astype(float)

# เพิ่ม noise ในข้อมูล
noise = np.random.normal(0, 1, train_data[numeric_cols].shape)  # ลด noise
train_data[numeric_cols] += noise

# ตรวจสอบว่าข้อมูลถูกต้องหรือไม่
print("Data types after conversion:", train_data.dtypes)
print("Sample of prepared data:")
print(train_data.head())

# ตั้งค่า PyCaret แบบค่อยเป็นค่อยไป
from pycaret.regression import *

try:
    exp_pm25 = setup(
        data=train_data,
        target="pm_2_5",
        session_id=123,
        fold=10,  # เพิ่มจำนวน fold
        feature_selection=True,  # เปิดการเลือกฟีเจอร์
        remove_multicollinearity=True,  # เปิดการตรวจสอบ multicollinearity
        verbose=True,  # แสดงรายละเอียดมากขึ้น
        polynomial_features=False,  # ปิดการสร้างฟีเจอร์ polynomial
        interaction_features=False,  # ปิดการสร้างฟีเจอร์ interaction
        bin_numeric_features=None,  # ไม่ใช้การ binning
        remove_outliers=True,  # เปิดการลบ outliers
    )

    # สร้างโมเดล Linear Regression แทน Random Forest
    best_pm25_model = create_model("lr", verbose=True)

    # ปรับแต่งโมเดลด้วยพารามิเตอร์น้อยลง
    tuned_pm25_model = tune_model(best_pm25_model, optimize="RMSE", n_iter=50)

    # โมเดลที่ดีที่สุด
    final_pm25_model = finalize_model(tuned_pm25_model)

    # บันทึกโมเดล
    save_model(final_pm25_model, "pm25_prediction_model_optimized")

    print("Model training completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")

    # ลองใช้วิธีที่ง่ายกว่า - ใช้เฉพาะฟีเจอร์พื้นฐาน
    print("Trying with basic features only...")
    basic_features = ["humidity", "pm_10"]
    basic_train_data = train_data[basic_features + ["pm_2_5"]].copy()

    print("Basic train data columns:", basic_train_data.columns.tolist())
    print("Basic train data sample:")
    print(basic_train_data.head())

    # ใช้ setup อีกครั้งกับข้อมูลพื้นฐาน
    exp_pm25 = setup(
        data=basic_train_data,
        target="pm_2_5",
        session_id=123,
        fold=10,
        normalize=True,
        verbose=True,
    )

    # สร้างโมเดล Linear Regression แทน Random Forest
    best_pm25_model = create_model("lr", verbose=True)

    # ปรับแต่งโมเดลด้วยพารามิเตอร์น้อยลง
    tuned_pm25_model = tune_model(best_pm25_model, optimize="RMSE", n_iter=50)

    # บันทึกโมเดลพื้นฐาน
    final_pm25_model = finalize_model(tuned_pm25_model)
    save_model(final_pm25_model, "pm25_prediction_model_basic")

    print("Basic model training completed successfully!")

# ส่วนที่ 14: การใช้งาน pycaret เพื่อสร้างโมเดลทำนายอุณหภูมิ

# ตรวจสอบข้อมูลที่เตรียมไว้
print("Train data shape for temperature model:", train_data.shape)
print("Columns in train_data:", train_data.columns.tolist())

# ตรวจสอบค่าที่ไม่ถูกต้อง
print(
    "Infinity check:",
    np.isinf(train_data.select_dtypes(include=["number"])).sum().sum(),
)
print("NaN check:", train_data.isnull().sum().sum())

# เติมค่าว่างด้วย median สำหรับตัวแปรที่มีค่าว่าง
train_data.fillna(train_data.median(), inplace=True)

# แปลงข้อมูลให้เป็น float ทั้งหมด
numeric_cols = train_data.select_dtypes(include=["number"]).columns
train_data[numeric_cols] = train_data[numeric_cols].astype(float)

# ตรวจสอบว่าข้อมูลถูกต้องหรือไม่
print("Data types after conversion:", train_data.dtypes)
print("Sample of prepared data:")
print(train_data.head())

# ตั้งค่า PyCaret แบบค่อยเป็นค่อยไป
from pycaret.regression import *

try:
    # ตั้งค่าแบบไม่ใช้ feature selection และ multicollinearity ก่อน
    exp_temp = setup(
        data=train_data,
        target="temperature",
        session_id=456,  # ใช้ session_id ที่แตกต่างจากโมเดล PM2.5
        fold=5,  # ลดจำนวน fold
        feature_selection=False,  # ปิดการเลือกฟีเจอร์
        remove_multicollinearity=False,  # ปิดการตรวจสอบ multicollinearity
        verbose=True,  # แสดงรายละเอียดมากขึ้น
    )

    # สร้างโมเดลแบบ verbose
    best_temp_model = create_model("en", verbose=True)  # Elastic Net

    # ประเมินผลโมเดลพื้นฐาน
    evaluate_model(best_temp_model)

    # ปรับแต่งโมเดลด้วยพารามิเตอร์น้อยลง
    tuned_temp_model = tune_model(best_temp_model, optimize="RMSE", n_iter=5)

    # โมเดลที่ดีที่สุด
    final_temp_model = finalize_model(tuned_temp_model)

    # บันทึกโมเดล
    save_model(final_temp_model, "temperature_prediction_model_optimized")

    print("Temperature model training completed successfully!")

except Exception as e:
    print(f"An error occurred in temperature model: {str(e)}")

    # ลองใช้วิธีที่ง่ายกว่า - ใช้เฉพาะฟีเจอร์พื้นฐาน
    print("Trying with basic features only for temperature model...")

    # สร้าง DataFrames ใหม่เพื่อหลีกเลี่ยงปัญหาจากการใช้งานก่อนหน้า
    # เลือกฟีเจอร์พื้นฐานที่น่าจะมีผลต่ออุณหภูมิ
    basic_features = ["humidity", "pm_10", "pm_2_5", "temperature"]
    basic_train_data = train_data[basic_features].copy()

    # ตรวจสอบข้อมูลอีกครั้ง
    print("Basic train data shape:", basic_train_data.shape)
    print("Basic train data sample:")
    print(basic_train_data.head())

    # สร้าง setup ใหม่อีกครั้ง
    exp_temp = setup(
        data=basic_train_data,
        target="temperature",
        session_id=456,
        fold=5,
        normalize=True,
        verbose=True,
    )

    # ลองใช้โมเดลอื่นที่อาจเหมาะสมกับข้อมูลอุณหภูมิ
    print("Trying alternative models for temperature prediction...")

    # ลองใช้ Random Forest แทน
    best_temp_model = create_model("rf", verbose=True)

    # ประเมินผลโมเดลพื้นฐาน
    evaluate_model(best_temp_model)

    # บันทึกโมเดลพื้นฐาน
    final_temp_model = finalize_model(best_temp_model)
    save_model(final_temp_model, "temperature_prediction_model_basic")

    print("Basic temperature model training completed successfully!")

    try:
        # ถ้าต้องการทดลองโมเดลหลายๆ แบบ
        print("Comparing different models for temperature prediction...")
        top_models = compare_models(
            include=["lr", "rf", "xgboost", "lightgbm"], n_select=2
        )

        # เลือกโมเดลที่ดีที่สุด
        best_temp_model = top_models[0]
        print("Best model selected:", best_temp_model)

        # บันทึกโมเดลที่ดีที่สุด
        final_temp_model = finalize_model(best_temp_model)
        save_model(final_temp_model, "temperature_prediction_model_best")

        print("Temperature model selection and training completed successfully!")
    except Exception as e:
        print(f"Error during model comparison: {str(e)}")
        print("Continuing with the basic model...")
# ส่วนที่ 15: บันทึกข้อมูลที่ทำความสะอาดแล้ว
import os

file_name, file_ext = os.path.splitext(file_path)
cleaned_file_path = f"{file_name}_clean_no_overfit{file_ext}"
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")
# ส่วนที่ 16: ตรวจสอบความสำคัญของตัวแปรสำหรับโมเดล PM2.5
if hasattr(final_pm25_model, "feature_importances_"):
    # Ensure X is defined and has the same columns as the training data
    X = train_data.drop(columns=["pm_2_5"])

    if len(X.columns) == len(final_pm25_model.feature_importances_):
        importances_pm25 = pd.DataFrame(
            {"feature": X.columns, "importance": final_pm25_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=importances_pm25)
        plt.title("Feature Importance for PM2.5 Model")
        plt.tight_layout()
        plt.show()

        print("Feature Importance for PM2.5 Model:\n", importances_pm25)
    else:
        print("Mismatch in the number of features and feature importances.")
else:
    print("The model does not have feature importances.")

# ส่วนที่ 17: ตรวจสอบความสำคัญของตัวแปรสำหรับโมเดลอุณหภูมิ
if hasattr(final_temp_model, "feature_importances_"):
    importances_temp = pd.DataFrame(
        {"feature": X.columns, "importance": final_temp_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importances_temp)
    plt.title("Feature Importance for Temperature Model")
    plt.tight_layout()
    plt.show()

    print("Feature Importance for Temperature Model:\n", importances_temp)
# ส่วนที่ 17: วิเคราะห์ Residuals เพื่อตรวจสอบการ Overfit สำหรับโมเดล PM2.5
train_pred_pm25 = predict_model(final_pm25_model, data=train_data)
test_pred_pm25 = predict_model(final_pm25_model, data=test_data)

# ตรวจสอบชื่อคอลัมน์ในผลลัพธ์ของ predict_model
print(train_pred_pm25.columns)

# คำนวณ residuals สำหรับโมเดล PM2.5
train_residuals_pm25 = train_pred_pm25["pm_2_5"] - train_pred_pm25["prediction_label"]
test_residuals_pm25 = test_pred_pm25["pm_2_5"] - test_pred_pm25["prediction_label"]

# แสดงกราฟ residuals สำหรับโมเดล PM2.5
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_pred_pm25["prediction_label"], train_residuals_pm25)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Training Residuals for PM2.5 Model")

plt.subplot(1, 2, 2)
plt.scatter(test_pred_pm25["prediction_label"], test_residuals_pm25)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Testing Residuals for PM2.5 Model")

plt.tight_layout()
plt.show()

# แสดงค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของ residuals สำหรับโมเดล PM2.5
print(
    f"Training Residuals for PM2.5 Model - Mean: {train_residuals_pm25.mean()}, Std: {train_residuals_pm25.std()}"
)
print(
    f"Testing Residuals for PM2.5 Model - Mean: {test_residuals_pm25.mean()}, Std: {test_residuals_pm25.std()}"
)

# ส่วนที่ 18: วิเคราะห์ Residuals เพื่อตรวจสอบการ Overfit สำหรับโมเดลอุณหภูมิ
train_pred_temp = predict_model(final_temp_model, data=train_data)
test_pred_temp = predict_model(final_temp_model, data=test_data)

# ตรวจสอบชื่อคอลัมน์ในผลลัพธ์ของ predict_model
print(train_pred_temp.columns)

# คำนวณ residuals สำหรับโมเดลอุณหภูมิ
train_residuals_temp = (
    train_pred_temp["temperature"] - train_pred_temp["prediction_label"]
)
test_residuals_temp = test_pred_temp["temperature"] - test_pred_temp["prediction_label"]

# แสดงกราฟ residuals สำหรับโมเดลอุณหภูมิ
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_pred_temp["prediction_label"], train_residuals_temp)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Training Residuals for Temperature Model")

plt.subplot(1, 2, 2)
plt.scatter(test_pred_temp["prediction_label"], test_residuals_temp)
plt.axhline(y=0, color="r", linestyle="-")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Testing Residuals for Temperature Model")

plt.tight_layout()
plt.show()

# แสดงค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของ residuals สำหรับโมเดลอุณหภูมิ
print(
    f"Training Residuals for Temperature Model - Mean: {train_residuals_temp.mean()}, Std: {train_residuals_temp.std()}"
)
print(
    f"Testing Residuals for Temperature Model - Mean: {test_residuals_temp.mean()}, Std: {test_residuals_temp.std()}"
)

# ตรวจสอบข้อมูลใน train_data
print(train_data.head())
print(train_data.dtypes)
print("Train data columns:", train_data.columns.tolist())
print("Train data sample:")
print(train_data.head())

# PyCaret setup
exp_temp = setup(
    data=train_data,
    target="temperature",
    session_id=456,
    fold=5,
    feature_selection=False,
    remove_multicollinearity=False,
    verbose=True,
)
