import streamlit as st
import pandas as pd
import numpy as np
import cv2
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os

# ---------- APP CONFIG ----------
st.set_page_config(page_title="Lunar Mission Dashboard", layout="wide")

st.title("🌕 Lunar Mission Dashboard")

# ---------- SIDEBAR NAV ----------
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an App Mode", ["Mission Outcome Prediction", "Lunar Landing Safety Analysis"])

# ---------- DB CONFIG ----------
DB_CREDENTIALS = {
    "host": "localhost",
    "database": "lunar_mission",
    "user": "postgres",
    "password": "admin",
    "port": 5432
}
engine = create_engine(
    f"postgresql://{DB_CREDENTIALS['user']}:{DB_CREDENTIALS['password']}@"
    f"{DB_CREDENTIALS['host']}:{DB_CREDENTIALS['port']}/{DB_CREDENTIALS['database']}"
)

def setup_database():
    try:
        conn = psycopg2.connect(**DB_CREDENTIALS)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mission_data (
            id SERIAL PRIMARY KEY,
            Nation TEXT,
            Type TEXT,
            Mission_Duration FLOAT,
            Launch_Year INT,
            Arrival_Year INT,
            Outcome TEXT
        );
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database setup failed: {e}")

setup_database()

# ---------- STATIC CSV LOAD ----------
def load_data():
    query = "SELECT * FROM mission_data"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise ValueError("Empty DB")
    except:
        try:
            st.info("Loading static CSV into PostgreSQL...")
            local_df = pd.read_csv("finalized_dataset_engineered.csv")
            local_df.to_sql("mission_data", engine, if_exists="replace", index=False)
            df = local_df
            st.success("Static CSV loaded into PostgreSQL.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            df = pd.DataFrame()
    return df

# ---------- LOAD DATA ----------
df = load_data()
if df.empty:
    st.warning("🚫 No data available.")
    st.stop()

# ---------- MISSION OUTCOME PREDICTION ----------
if app_mode == "Mission Outcome Prediction":
    st.header("🚀 Mission Outcome Prediction")

    df['Outcome_Binary'] = df['Outcome'].apply(lambda x: 1 if 'Successful' in str(x) else 0)
    features = ['Nation', 'Type', 'Mission_Duration', 'Launch_Year', 'Arrival_Year']
    X = pd.get_dummies(df[features], columns=['Nation', 'Type'], drop_first=True)
    y = df['Outcome_Binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_options = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Neural Network": None
    }
    selected_model = st.selectbox("🔍 Select a Model", list(model_options.keys()))

    if selected_model != "Neural Network":
        model = model_options[selected_model]
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.write(f"✅ Accuracy: **{accuracy:.2f}**")
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        nn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
        accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        st.write(f"✅ Neural Network Accuracy: **{accuracy:.2f}**")

    # ---------- USER INTERACTIVITY ----------
    unique_nations = df['Nation'].unique()
    nation = st.selectbox("Nation", unique_nations)
    filtered_df = df[df['Nation'] == nation]
    if not filtered_df.empty:
        mission_type = st.selectbox("Type", filtered_df['Type'].unique())
        filtered_df_type = filtered_df[filtered_df['Type'] == mission_type]
        if not filtered_df_type.empty:
            launch_year = st.selectbox("Launch Year", sorted(filtered_df_type['Launch_Year'].dropna().unique()))
            arrival_years_for_launch = sorted(filtered_df_type[filtered_df_type['Launch_Year'] == launch_year]['Arrival_Year'].dropna().unique())
            if arrival_years_for_launch:
                arrival_year = st.selectbox("Arrival Year", arrival_years_for_launch)
                durations_for_launch_arrival = sorted(filtered_df_type[(filtered_df_type['Launch_Year'] == launch_year) & (filtered_df_type['Arrival_Year'] == arrival_year)]['Mission_Duration'].dropna().unique())
                if durations_for_launch_arrival:
                    mission_duration = st.selectbox("Mission Duration (days)", durations_for_launch_arrival)

                    input_data = pd.DataFrame({
                        'Nation': [nation],
                        'Type': [mission_type],
                        'Launch_Year': [launch_year],
                        'Arrival_Year': [arrival_year],
                        'Mission_Duration': [mission_duration]
                    })
                    input_data_encoded = pd.get_dummies(input_data, columns=['Nation', 'Type'], drop_first=True)
                    for col in X_train.columns:
                        if col not in input_data_encoded.columns:
                            input_data_encoded[col] = 0
                    input_data_encoded = input_data_encoded[X_train.columns]

                    if st.button("Predict Outcome"):
                        if selected_model != "Neural Network":
                            prediction = model.predict(input_data_encoded)
                            st.success(f"{selected_model} Predicted Outcome: {'Successful' if prediction[0] == 1 else 'Unsuccessful'}")
                        else:
                            input_scaled = scaler.transform(input_data_encoded)
                            prediction = nn_model.predict(input_scaled)
                            st.success(f"Neural Network Predicted Outcome: {'Successful' if prediction[0][0] > 0.5 else 'Unsuccessful'}")

    # ---------- BENCHMARK COMPARISON ----------
    st.subheader("📊 Model Accuracy Comparison")
    results = {}
    for name, clf in model_options.items():
        if name == "Neural Network":
            results[name] = accuracy
        else:
            clf.fit(X_train, y_train)
            results[name] = clf.score(X_test, y_test)

    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(), color=['skyblue', 'orange', 'limegreen', 'violet'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    st.pyplot(fig)

# ---------- LUNAR LANDING SAFETY ----------

st.title("🌑 Enhanced Lunar Landing Safety Analysis")

...
# The rest of your lunar landing safety analysis code remains unchanged
...



def analyze_landing_safety(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    normalized = resized / 255.0

    # Crater Detection
    circles = cv2.HoughCircles(resized, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=50)
    output_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(output_image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    # Surface Feature Analysis
    laplacian_var = cv2.Laplacian(normalized, cv2.CV_64F).var()
    sobelx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    avg_slope = np.mean(gradient_magnitude)

    # Safety Score
    safety_score = (1 / (1 + laplacian_var + avg_slope)) * 100
    num_craters = len(circles[0]) if circles is not None else 0

    return output_image, safety_score, num_craters, laplacian_var, avg_slope, gradient_magnitude


def generate_hazard_heatmap(image, gradient_magnitude):
    heatmap = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    return overlay


def estimate_sun_angle(normalized):
    sobelx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=5)
    mean_direction = np.mean(sobelx)
    angle = np.arctan(mean_direction) * (180 / np.pi)
    return angle


def overlay_trajectory(image):
    h, w = image.shape[:2]
    trajectory_image = image.copy()
    cv2.line(trajectory_image, (w//2, 0), (w//2, h), (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(5, h, 40):
        cv2.circle(trajectory_image, (w//2, i), 5, (255, 255, 255), -1)
    return trajectory_image


def classify_patch_features(lap_var, slope, crater):
    if lap_var < 0.005 and slope < 0.01:
        return "Smooth"
    elif crater > 2:
        return "Cratered"
    elif slope > 0.02:
        return "Rocky"
    else:
        return "Moderate"


uploaded_files = st.file_uploader("📷 Upload lunar surface images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("🖼️ Enhanced Analysis Results")
    results = []

    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            out_img, score, craters, lap_var, slope, grad_mag = analyze_landing_safety(image)
            score_color = "🟢" if score > 70 else "🟠" if score > 50 else "🔴"
            sun_angle = estimate_sun_angle(image / 255.0)
            hazard_map = generate_hazard_heatmap(out_img, grad_mag)
            trajectory_img = overlay_trajectory(out_img)
            terrain_type = classify_patch_features(lap_var, slope, craters)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), caption=f"{score_color} Safety Score: {score:.2f}", use_column_width=True)
                st.image(cv2.cvtColor(trajectory_img, cv2.COLOR_BGR2RGB), caption="🧭 Descent Trajectory", use_column_width=True)
            with col2:
                st.image(cv2.cvtColor(hazard_map, cv2.COLOR_BGR2RGB), caption="🔥 Hazard Zoning Heatmap", use_column_width=True)

            with st.expander("📋 Detailed Metrics"):
                st.markdown(f"- **File:** `{uploaded_file.name}`")
                st.markdown(f"- **Crater Count**: {craters}")
                st.markdown(f"- **Laplacian Variance**: {lap_var:.4f}")
                st.markdown(f"- **Average Slope**: {slope:.4f}")
                st.markdown(f"- **Estimated Sun Angle**: {sun_angle:.2f}°")
                st.markdown(f"- **Terrain Type**: {terrain_type}")

            results.append({
                "Filename": uploaded_file.name,
                "Safety Score": round(score, 2),
                "Crater Count": craters,
                "Laplacian Variance": round(lap_var, 4),
                "Average Slope": round(slope, 4),
                "Sun Angle": round(sun_angle, 2),
                "Terrain Type": terrain_type
            })
        else:
            st.error(f"❌ Unable to process image: {uploaded_file.name}")

    if results:
        df_results = pd.DataFrame(results)
        st.markdown("### 🧾 Analysis Summary Table")
        st.dataframe(df_results)

        st.markdown("### 📊 Safety Score Chart")
        fig, ax = plt.subplots()
        ax.bar(df_results["Filename"], df_results["Safety Score"], color='skyblue')
        ax.set_ylabel("Safety Score")
        ax.set_ylim([0, 100])
        ax.set_xticklabels(df_results["Filename"], rotation=45, ha='right')
        st.pyplot(fig)

        if st.checkbox("💾 Save results to PostgreSQL"):
            try:
                df_results.to_sql("enhanced_landing_safety_results", engine, if_exists="append", index=False)
                st.success("✅ Results saved to database.")
            except Exception as e:
                st.error(f"❌ Failed to save to database: {e}")
