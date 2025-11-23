import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import onnxruntime as ort



df = pd.read_csv('drone_delivery.csv')

df['ETA_actual_min'] = (
    (df['distance_km'] / df['drone_speed_kmph']) * 60 +  # base ETA
    np.random.uniform(0, 10) +                   # random noise
    np.where(df['climate_condition'] == 'Rainy', 5, 0) +  # weather penalty
    np.where(df['payload_weight_kg'] > 3, 4, 0)                  # payload penalty
).round(2)

# Fill NaN values in ETA_actual_min before proceeding
df['ETA_actual_min'].fillna(df['ETA_actual_min'].mean(), inplace=True)

# Encode categorical features
le_type = LabelEncoder()
le_climate = LabelEncoder()

df['drone_type_encoded'] = le_type.fit_transform(df['drone_type'])
df['climate_encoded'] = le_climate.fit_transform(df['climate_condition'])

# Features and Target
X = df[['distance_km', 'payload_weight_kg', 'drone_speed_kmph', 'battery_efficiency','drone_type_encoded', 
        'climate_encoded']]
y = df['ETA_actual_min']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

print(df['ETA_actual_min'].isna().sum())

missing_count = df['ETA_actual_min'].isna().sum()
print(f"Dropped {missing_count} rows with missing ETA_actual values.")

df = df.dropna(subset=['ETA_actual_min'])

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic drone delivery dataset
n = 1000

drone_types = ['Quadcopter', 'Hexacopter', 'Octocopter']
weather_conditions = ['Clear', 'Rainy', 'Windy', 'Foggy']

data = {
    'order_id': [f'ORD_{i+1}' for i in range(n)],
    'drone_id': [f'DRN_{np.random.randint(100, 999)}' for _ in range(n)],
    'drone_type': np.random.choice(drone_types, n),
    'distance_km': np.random.uniform(1, 50, n).round(2),

    'payload_kg': np.random.uniform(0.5, 5.0, n).round(2),
    'drone_speed': np.random.uniform(20, 60, n).round(2),
    'weather_condition': np.random.choice(weather_conditions, n),
    'battery_efficiency': np.random.uniform(60, 100, n).round(2)
}

df = pd.DataFrame(data)

# Simulate ETA (target variable)
df['ETA_actual'] = (
    (df['distance_km'] / df['drone_speed']) * 60 +  # base ETA
    np.random.uniform(0, 10, n) +                   # random noise
    np.where(df['weather_condition'] == 'Rainy', 5, 0) +  # weather penalty
    np.where(df['payload_kg'] > 3, 4, 0)                  # payload penalty
).round(2)

# Encode categorical features
le_type = LabelEncoder()
le_weather = LabelEncoder()

df['drone_type_encoded'] = le_type.fit_transform(df['drone_type'])
df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])

# Features and Target
X = df[['distance_km', 'payload_kg', 'drone_speed', 'battery_efficiency', 
        'drone_type_encoded', 'weather_encoded']]
y = df['ETA_actual']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel('Actual ETA (min)')
plt.ylabel('Predicted ETA (min)')
plt.title('Actual vs Predicted ETA')
plt.grid(True)
plt.show()

joblib.dump(model, 'eta_prediction_model.pkl')
print("✅ Model saved as eta_prediction_model.pkl")

# Load dataset
df = pd.read_csv("D:\data science\Aaro ai\drone_eta_dataset.csv")

num_cols = df.select_dtypes(include=['int64','float64']).columns

df[num_cols].hist(figsize=(12,8))
plt.tight_layout()
plt.show()
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()
    
    plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.show()
sns.scatterplot(data=df, x="distance_km", y="ETA_actual", hue="drone_type")
plt.show()
sns.scatterplot(data=df, x="payload_kg", y="ETA_actual", hue="drone_type")
plt.show()
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, y=col)
    plt.title(col)
    plt.show()
scaler = StandardScaler()
scale_cols = ['distance_km', 'payload_kg', 'drone_speed', 'battery_efficiency']

df[scale_cols] = scaler.fit_transform(df[scale_cols])
df.to_csv("cleaned_drone_eta.csv", index=False)

# ---------------------------------------------------------
# STREAMLIT APP – Drone ETA Prediction
# ------------------------------
# Load Model + Dataset
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("D:\data science\Aaro ai\eta_prediction_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("D:\data science\Aaro ai\yolov5\cleaned_drone_eta.csv")

model = load_model()
df = load_data()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Drone ETA Prediction", page_icon="🚁", layout="wide")

st.title("🚁 Drone Delivery ETA Prediction")
st.write("Predict Estimated Delivery Time based on drone metrics.")

st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Input Parameters")

# Prepare options
drone_types = df["drone_type"].unique()
weather_types = df["weather_encoded"].unique()

# ------------------------------
# Input Form
# ------------------------------
distance = st.sidebar.number_input("Distance (km)", min_value=1.0, max_value=50.0, step=0.5)
payload = st.sidebar.number_input("Payload Weight (kg)", min_value=0.1, max_value=5.0, step=0.1)
speed = st.sidebar.number_input("Drone Speed (km/h)", min_value=10.0, max_value=60.0, step=1.0)
battery = st.sidebar.number_input("Battery Efficiency (%)", min_value=50.0, max_value=100.0, step=1.0)

drone_type = st.sidebar.selectbox("Drone Type (Encoded)", sorted(drone_types))
weather = st.sidebar.selectbox("Weather Condition (Encoded)", sorted(weather_types))

predict_btn = st.sidebar.button("Predict ETA ⏱️")

# ------------------------------
# Prediction Logic
# ------------------------------
if predict_btn:
    input_data = np.array([[distance, payload, speed, battery, drone_type, weather]])
    eta = model.predict(input_data)[0]

    st.success(f"### Predicted ETA: **{eta:.2f} minutes**")
    st.balloons()

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

# Config / File paths
# -------------------
DATA_PATH = "/mnt/data/cleaned_drone_eta.csv"
MODEL_PATH = "/mnt/data/eta_model.pkl"
YOLO_ONNX_PATH = "/mnt/data/yolov8m.onnx"
YOLO_PT_PATH = "/mnt/data/best.pt"
REPORTS_DIR = "/mnt/data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------
# Utilities
# -------------------
@st.cache_data
def load_data(path=DATA_PATH):
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_model(path=MODEL_PATH):
    if Path(path).exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model from {path}: {e}")
            return None
    return None

def save_pdf_report(rows: pd.DataFrame, report_name: str):
    report_path = Path(REPORTS_DIR) / report_name
    doc = SimpleDocTemplate(str(report_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Aero AI - ETA Prediction Report", styles['Title']))
    story.append(Spacer(1,12))
    for _, r in rows.iterrows():
        txt = f"Order: {r.get('order_id', '')} — Drone: {r.get('drone_id','')}"
        story.append(Paragraph(txt, styles['Heading3']))
        story.append(Paragraph(f"Distance: {r.get('distance_km', '')} km, Payload: {r.get('payload_kg','')} kg", styles['BodyText']))
        story.append(Paragraph(f"Predicted ETA: {r.get('ETA_predicted','')} min, Actual ETA: {r.get('ETA_actual','')}", styles['BodyText']))
        story.append(Spacer(1,10))
    doc.build(story)
    return str(report_path)

def make_prediction_input(distance, payload, speed, battery, drone_type_encoded, weather_encoded):
    # Adjust order of features to match training; update if needed
    return np.array([[distance, payload, speed, battery, drone_type_encoded, weather_encoded]], dtype=float)

# Simple weather map (update to match your preprocessing mapping)
WEATHER_MAP = {"Clear":0, "Cloudy":1, "Rainy":2, "Windy":3, "Storm":4}

# -------------------
# Page: Home
# -------------------
st.set_page_config(page_title="Aero AI", layout="wide")
st.sidebar.title("Aero AI — Navigation")
page = st.sidebar.radio("Go to", ["Home", "ETA Prediction", "Drone Detection", "Chatbot / Reports", "About"])

df = load_data()
model = load_model()

if page == "Home":
    st.title("🚁 Aero AI — Drone Delivery System")
    st.write("Project: Drone ETA prediction, Drone detection (CV), multi-agent chatbot and report generation.")
    st.markdown("**Dataset path:** `" + DATA_PATH + "`")
    st.markdown("**Model path:** `" + MODEL_PATH + "` (place model here or use upload in ETA page)")
    st.markdown("---")
    st.subheader("Quick dataset preview")
    if df.empty:
        st.info("Dataset not found at `/mnt/data/cleaned_drone_eta.csv`. Upload or place the file there.")
        uploaded = st.file_uploader("Or upload cleaned_drone_eta.csv here", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("Uploaded dataset loaded to app memory. You can save it to `/mnt/data/` manually.")
    else:
        st.dataframe(df.head())
        st.write(df.describe(include='all'))

# -------------------
# Page: ETA Prediction
# -------------------
elif page == "ETA Prediction":
    st.header("ETA Prediction")
    st.markdown("Use the model to predict ETA for a single delivery or batch predict a portion of dataset.")
    # show model status
    if model is None:
        st.warning(f"No model loaded from `{MODEL_PATH}`. Upload a model file (.pkl) below or train and save one as joblib/pickle to that path.")
        uploaded_model = st.file_uploader("Upload eta_model.pkl", type=["pkl","joblib","sav"], key="upload_model")
        if uploaded_model:
            try:
                joblib.dump(joblib.load(uploaded_model), MODEL_PATH)  # attempt to save uploaded model bytes
            except Exception:
                with open(MODEL_PATH, "wb") as f:
                    f.write(uploaded_model.getvalue())
            model = load_model()
            st.success("Model uploaded and loaded.")
    else:
        st.success("Model loaded.")

    # Input form
    with st.form("single_predict"):
        st.subheader("Single prediction")
        col1, col2 = st.columns(2)
        distance = col1.number_input("Distance (km)", min_value=0.1, value=5.0, step=0.1)
        payload = col1.number_input("Payload (kg)", min_value=0.0, value=1.0, step=0.1)
        speed = col2.number_input("Drone Speed (km/h)", min_value=1.0, value=40.0, step=1.0)
        battery = col2.number_input("Battery Efficiency (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        # encoded options fallback: if dataset has encoded columns show choices else provide mapping
        if "drone_type" in df.columns:
            types = sorted(df["drone_type"].unique().tolist())
            drone_type = st.selectbox("Drone Type (encoded)", types)
            drone_type_encoded = int(drone_type)
        else:
            drone_type_encoded = 0
        weather = st.selectbox("Weather", list(WEATHER_MAP.keys()))
        weather_encoded = WEATHER_MAP.get(weather, 0)
        submit = st.form_submit_button("Predict")
    if submit:
        if model is None:
            st.error("Model not loaded — place model at `/mnt/data/eta_model.pkl` or upload above.")
        else:
            X = make_prediction_input(distance, payload, speed, battery, drone_type_encoded, weather_encoded)
            try:
                pred = model.predict(X)[0]
                st.success(f"Predicted ETA: **{pred:.2f} minutes**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Batch predictions on dataset")
    if not df.empty and model is not None:
        n_rows = st.slider("How many rows to predict from dataset head", 1, min(500, len(df)), 10)
        sample = df.head(n_rows).copy()
        # Prepare features — this block assumes specific column names: adjust based on your training X
        try:
            # attempt to compute or use encoded columns if exist
            def encode_row(r):
                # flexible: handle different column names
                dist = float(r.get("distance_km", r.get("distance", 0)))
                payload = float(r.get("payload_kg", r.get("payload_weight_kg", 0)))
                speed = float(r.get("drone_speed", r.get("drone_speed_kmph", 0)))
                battery = float(r.get("battery_efficiency", r.get("battery_level", 90)))
                dt = int(r.get("drone_type", 0))
                wc = r.get("weather_condition", None)
                wc_enc = WEATHER_MAP.get(wc, r.get("weather_encoded", 0) if "weather_encoded" in r else 0)
                return [dist, payload, speed, battery, dt, wc_enc]
            X_batch = np.array([encode_row(r) for _, r in sample.iterrows()], dtype=float)
            preds = model.predict(X_batch)
            sample["ETA_predicted"] = np.round(preds, 2)
            st.dataframe(sample.head(50))
            # offer download of result CSV
            csv = sample.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="eta_predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed; check dataset column names & training feature order. Error: {e}")
    elif df.empty:
        st.info("No dataset loaded to run batch predictions.")

# -------------------
# Page: Drone Detection
# -------------------
elif page == "Drone Detection":
    st.header("Drone Detection (Image upload)")
    st.markdown("Upload an image and run detection. If you have ONNX model (`/mnt/data/yolov8m.onnx`) or YOLO weights (`/mnt/data/best.pt`), the app will attempt to use them.")
    image_file = st.file_uploader("Upload drone image", type=["jpg","jpeg","png"])
    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        # Try ONNX detection (lightweight). Fallback: show message.
        if Path(YOLO_ONNX_PATH).exists():
            st.info("ONNX model found; running ONNX inference (requires onnxruntime).")
            try:
                import onnxruntime as ort
                import cv2
                # Prepare image
                arr = np.array(img)
                h, w = arr.shape[:2]
                # Basic preproc: resize to 640; adjust as needed depending on how model expects input
                import torchvision.transforms as T
                transform = T.Compose([T.ToPILImage(), T.Resize((640,640)), T.ToTensor()])
                tensor = transform(arr).unsqueeze(0).numpy()
                sess = ort.InferenceSession(YOLO_ONNX_PATH)
                # ONNX inputs/outputs differ by model. This is a placeholder and may need model-specific mapping.
                input_name = sess.get_inputs()[0].name
                out = sess.run(None, {input_name: tensor})
                st.write("Inference completed (raw). You may need to adapt preprocessing/postprocessing for your exact ONNX model.")
            except Exception as e:
                st.error(f"ONNX inference failed: {e}")
        elif Path(YOLO_PT_PATH).exists():
            st.info("YOLO .pt weights found; attempting inference via ultralytics (if installed).")
            try:
                from ultralytics import YOLO
                model_det = YOLO(str(YOLO_PT_PATH))
                results = model_det(np.array(img))
                # show first result image with boxes
                res_img = results[0].plot()
                st.image(res_img, caption="Detection result", use_column_width=True)
            except Exception as e:
                st.error(f"YOLO .pt inference failed: {e}\nMake sure ultralytics is installed and GPU available or use CPU mode.")
        else:
            st.warning("No detection model found at /mnt/data/yolov8m.onnx or /mnt/data/best.pt. Upload or place model file to enable detection.")

# -------------------
# Page: Chatbot / Reports
# -------------------
elif page == "Chatbot / Reports":
    st.header("Chatbot & Report Generation")
    st.markdown("Ask simple dataset questions (e.g., 'top 5 longest deliveries', 'average ETA by drone_type') and generate PDF reports.")
    # Simple keyword-based query engine using dataframe
    query = st.text_input("Ask dataset question (examples: 'avg ETA', 'top 5 longest deliveries', 'count by drone_type')")
    if st.button("Run Query"):
        if df.empty:
            st.error("Dataset not loaded. Please upload `cleaned_drone_eta.csv` to /mnt/data or use Home page uploader.")
        else:
            q = query.lower()
            if "avg" in q or "average" in q:
                st.write("Average ETA_actual:", df["ETA_actual"].mean())
            elif "top" in q and "long" in q:
                st.write(df.sort_values("distance_km", ascending=False).head(10))
            elif "count" in q and "drone_type" in q:
                st.write(df["drone_type"].value_counts())
            else:
                st.write("Query not recognized. Try examples: 'avg ETA', 'top 5 longest deliveries', 'count by drone_type'")

    st.markdown("---")
    st.subheader("Generate PDF report from selected rows")
    use_index = st.number_input("Start index (0-based)", min_value=0, max_value=max(0, len(df)-1), value=0)
    num = st.number_input("Number of rows", min_value=1, max_value=min(100, max(1, len(df)-use_index)), value=5)
    if st.button("Generate PDF Report"):
        if df.empty:
            st.error("No data to generate report.")
        else:
            rows = df.iloc[use_index:use_index+num]
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"report_{now}.pdf"
            path = save_pdf_report(rows, report_name)
            st.success(f"Report generated: {path}")
            with open(path, "rb") as f:
                st.download_button("Download Report", data=f, file_name=report_name)

# -------------------
# Page: About
# -------------------
elif page == "About":
    st.header("About Aero AI Project")
    st.markdown("""
    **Aero AI**: Drone Delivery ETA Prediction + Drone Detection + Chatbot + Report generation.
    - Dataset path: `/mnt/data/cleaned_drone_eta.csv`
    - ETA model: `/mnt/data/eta_model.pkl`
    - Place your detection model (onnx or .pt) in `/mnt/data/` for detection page.
    """)
    st.markdown("Created as submission-ready project.")
