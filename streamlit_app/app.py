import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import joblib
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Streamlit App Config
st.set_page_config(page_title="Lunar Mission Dashboard", layout="wide")
st.title("🌕 Lunar Mission Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an App Mode", ["Mission Outcome Prediction", "Lunar Landing Safety Analysis"])

# ------------------------ Mission Outcome Prediction ------------------------
if app_mode == "Mission Outcome Prediction":
    st.header("Mission Outcome Prediction")

    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("finalized_dataset_engineered.csv")
            return df
        except FileNotFoundError:
            st.error("Dataset file not found. Please check the file path.")
            return None

    df = load_data()
    if df is not None:
        if st.checkbox("Show Raw Data"):
            st.write(df)

        # Feature Engineering
        df['Outcome_Binary'] = df['Outcome'].apply(lambda x: 1 if 'Successful' in str(x) else 0)
        features = ['Nation', 'Type', 'Mission_Duration', 'Launch_Year', 'Arrival_Year']
        X = pd.get_dummies(df[features], columns=['Nation', 'Type'], drop_first=True)
        y = df['Outcome_Binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_options = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42),
            "Neural Network": None
        }
        selected_model = st.selectbox("Select a Model", list(model_options.keys()))

        if selected_model != "Neural Network":
            model = model_options[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))
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
            y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int)
            st.write("Accuracy:", accuracy_score(y_test, y_pred_nn))
            st.text(classification_report(y_test, y_pred_nn))

        # Step 10: User Input for Prediction
        st.subheader("Make a Prediction")

        # Get unique values from the dataset for user input
        unique_nations = df['Nation'].unique()

        # User input fields based on dataset
        nation = st.selectbox("Nation", unique_nations)

        # Filter dataset based on selected nation
        filtered_df = df[df['Nation'] == nation]

        # Check if filtered_df is empty
        if filtered_df.empty:
            st.warning("No data available for the selected Nation.")
        else:
            # Get unique values for the filtered dataset
            unique_types = filtered_df['Type'].unique()

            # Dropdown for Type
            mission_type = st.selectbox("Type", unique_types)

            # Filter dataset based on selected nation and type
            filtered_df_type = filtered_df[filtered_df['Type'] == mission_type]

            # Check if filtered_df_type is empty
            if filtered_df_type.empty:
                st.warning("No data available for the selected Nation and Type.")
            else:
                # Get unique values for the filtered dataset (based on nation and type)
                unique_launch_years = sorted(filtered_df_type['Launch_Year'].dropna().unique())

                # Dropdown for Launch Year
                launch_year = st.selectbox("Launch Year", unique_launch_years)

                # Filter Arrival Years based on selected Launch Year
                arrival_years_for_launch = sorted(filtered_df_type[filtered_df_type['Launch_Year'] == launch_year]['Arrival_Year'].dropna().unique())

                # Check if arrival_years_for_launch is empty
                if not arrival_years_for_launch:
                    st.warning("No data available for the selected Launch Year.")
                else:
                    arrival_year = st.selectbox("Arrival Year", arrival_years_for_launch)

                    # Filter Mission Durations based on selected Launch Year and Arrival Year
                    durations_for_launch_arrival = sorted(filtered_df_type[(filtered_df_type['Launch_Year'] == launch_year) & (filtered_df_type['Arrival_Year'] == arrival_year)]['Mission_Duration'].dropna().unique())

                    # Check if durations_for_launch_arrival is empty
                    if not durations_for_launch_arrival:
                        st.warning("No data available for the selected Launch Year and Arrival Year.")
                    else:
                        mission_duration = st.selectbox("Mission Duration (days)", durations_for_launch_arrival)

                        # Prepare user input for prediction
                        input_data = pd.DataFrame({
                            'Nation': [nation],
                            'Type': [mission_type],
                            'Launch_Year': [launch_year],
                            'Arrival_Year': [arrival_year],
                            'Mission_Duration': [mission_duration]
                        })

                        # One-hot encode user input
                        input_data_encoded = pd.get_dummies(input_data, columns=['Nation', 'Type'], drop_first=True)

                        # Ensure the input data has the same columns as the training data
                        for col in X_train.columns:
                            if col not in input_data_encoded.columns:
                                input_data_encoded[col] = 0

                        # Reorder columns to match the training data
                        input_data_encoded = input_data_encoded[X_train.columns]

                        # Input Validation Function
                        def validate_input(input_data):
                            """
                            Validate user input to ensure it is within the expected range or format.
                            """
                            # Check if Launch Year is valid
                            if input_data['Launch_Year'].iloc[0] not in unique_launch_years:
                                return False, "Incorrect Data: Invalid Launch Year"
                            
                            # Check if Arrival Year is valid
                            if input_data['Arrival_Year'].iloc[0] not in arrival_years_for_launch:
                                return False, "Incorrect Data: Invalid Arrival Year"
                            
                            # Check if Mission Duration is valid
                            if input_data['Mission_Duration'].iloc[0] not in durations_for_launch_arrival:
                                return False, "Incorrect Data: Invalid Mission Duration"
                            
                            # If all checks pass, return True
                            return True, "Valid Data"

                        # Make a prediction
                        if st.button("Predict Outcome"):
                            # Validate user input
                            is_valid, validation_message = validate_input(input_data)
                            
                            if not is_valid:
                                st.error(validation_message)  # Display error message for incorrect data
                            else:
                                if selected_model != "Neural Network":
                                    prediction = model.predict(input_data_encoded)
                                    st.success(f"{selected_model} Predicted Outcome: {'Successful' if prediction[0] == 1 else 'Unsuccessful'}")
                                else:
                                    input_data_scaled = scaler.transform(input_data_encoded)
                                    nn_prediction = (nn_model.predict(input_data_scaled) > 0.5).astype(int)
                                    st.success(f"Neural Network Predicted Outcome: {'Successful' if nn_prediction[0] == 1 else 'Unsuccessful'}")

# ------------------------ Lunar Landing Safety Analysis ------------------------
elif app_mode == "Lunar Landing Safety Analysis":
    st.header("Lunar Landing Safety Analysis")

    def analyze_landing_safety(image):
        """
        Process lunar surface images to evaluate landing safety using OpenCV.
        """
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (256, 256))
            normalized = resized / 255.0  # Normalize pixel values

            # Detect craters using Hough Circle Transform
            circles = cv2.HoughCircles(
                resized,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=50
            )

            # Draw detected craters on the image
            output_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    cv2.circle(output_image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)  # Draw circle
                    cv2.circle(output_image, (circle[0], circle[1]), 2, (0, 0, 255), 3)  # Draw center

            # Calculate terrain roughness using Laplacian and Sobel
            laplacian_var = cv2.Laplacian(normalized, cv2.CV_64F).var()
            sobelx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            avg_slope = np.mean(gradient_magnitude)

            # Calculate safety score
            safety_score = (1 / (1 + laplacian_var + avg_slope)) * 100
            safety_label = "Safe" if safety_score > 50 else "Unsafe"

            # Count the number of craters
            num_craters = len(circles[0]) if circles is not None else 0

            return output_image, safety_score, safety_label, num_craters, laplacian_var, avg_slope

        except Exception as e:
            return None, None, f"Error processing image: {str(e)}", None, None, None

    # File uploader for lunar images
    uploaded_files = st.file_uploader("Upload lunar images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as BGR image for processing

            if image is not None:
                # Analyze landing safety
                output_image, safety_score, safety_label, num_craters, laplacian_var, avg_slope = analyze_landing_safety(image)

                if output_image is not None:
                    # Display processed image
                    st.image(output_image, caption=f"Safety Score: {safety_score:.2f} - {safety_label}", use_column_width=True)

                    # Add results to the list
                    results.append({
                        "filename": uploaded_file.name,
                        "safety_score": safety_score,
                        "safety_label": safety_label,
                        "num_craters": num_craters,
                        "laplacian_var": laplacian_var,
                        "avg_slope": avg_slope
                    })
                else:
                    st.error("Image processing failed. Please upload a valid lunar surface image.")
            else:
                st.error("Failed to decode uploaded image. Please try again.")

        # Display analysis results in a table
        if results:
            st.header("Analysis Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            # Visualizations
            st.header("Visualizations")

            # Safety Score Distribution
            st.write("### Safety Score Distribution")
            fig, ax = plt.subplots()
            ax.hist(results_df["safety_score"], bins=10, color='blue', alpha=0.7)
            ax.set_xlabel("Safety Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Safety Score Distribution")
            st.pyplot(fig)

            # Number of Craters vs. Safety Score
            st.write("### Number of Craters vs. Safety Score")
            fig, ax = plt.subplots()
            ax.scatter(results_df["num_craters"], results_df["safety_score"])
            ax.set_xlabel("Number of Craters")
            ax.set_ylabel("Safety Score")
            ax.set_title("Number of Craters vs. Safety Score")
            st.pyplot(fig)

            # Terrain Roughness Metrics
            st.write("### Terrain Roughness Metrics")
            fig, ax = plt.subplots()
            ax.bar(results_df["filename"], results_df["laplacian_var"], label="Laplacian Variance")
            ax.bar(results_df["filename"], results_df["avg_slope"], label="Average Slope", alpha=0.7)
            ax.set_xlabel("Image Filename")
            ax.set_ylabel("Value")
            ax.set_title("Terrain Roughness Metrics")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.info("Please upload lunar images to get started.")