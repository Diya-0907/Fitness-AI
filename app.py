import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# ------------------ LOAD MODELS ------------------

workout_model = joblib.load("models/workout_recommendation_model.pkl")
food_calorie_model = joblib.load("models/indian_food_calorie_model.pkl")
food_image_model = tf.keras.models.load_model("models/food_image_classification_model.h5")

# ------------------ UI ------------------

st.set_page_config(page_title="AI Health & Food Dashboard", layout="wide")
st.title("🤖 AI Health & Food Models")

col1, col2, col3 = st.columns(3)

# ------------------ WORKOUT MODEL ------------------
with col1:
    st.subheader("🏋️ Workout Recommendation")

    age = st.number_input("Age", min_value=10, max_value=80)
    weight = st.number_input("Weight (kg)")
    height = st.number_input("Height (cm)")
    goal = st.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Maintenance"])

    goal_map = {
        "Weight Loss": 0,
        "Muscle Gain": 1,
        "Maintenance": 2
    }

    workout_map = {
        0: "Cardio",
        1: "Strength Training",
        2: "Yoga & Flexibility"
    }

    if st.button("Predict Workout"):
        X = np.zeros((1, 11))
        X[0][0] = age
        X[0][1] = weight
        X[0][2] = height
        X[0][3] = goal_map[goal]

        prediction = workout_model.predict(X)[0]

        st.success(
            f"Recommended Workout: {workout_map.get(prediction, 'Custom Plan')}"
        )




# ------------------ FOOD CALORIE MODEL ------------------
with col2:
    st.subheader("🍛 Indian Food Calorie Prediction")

    food_name = st.text_input("Enter food name")

    if st.button("Predict Calories"):
        calorie = food_calorie_model.predict([food_name])[0]
        st.success(f"Estimated Calories: {int(calorie)} kcal")

# ------------------ FOOD IMAGE CLASSIFICATION ------------------
with col3:
    st.subheader("🍔 Food Image Classification")

    image_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((224, 224))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Classify Image"):
            preds = food_image_model.predict(img_array)[0]

            best_idx = int(np.argmax(preds))
            best_conf = float(preds[best_idx])

            CONFIDENCE_THRESHOLD = 0.6

            if best_conf < CONFIDENCE_THRESHOLD:
                detected_food = "Food detected (not confidently recognized)"
            else:
                detected_food = f"Closest known food class: {best_idx}"

            st.success(detected_food)
            st.caption(f"Confidence score: {best_conf:.2f}")
            st.bar_chart(preds)


