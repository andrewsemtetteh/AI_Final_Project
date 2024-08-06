import streamlit as st
import pandas as pd
import numpy as np
import os
import replicate
from langchain.prompts import PromptTemplate
from langchain.llms import Replicate
from langchain.chains import LLMChain
import math

# loading our API token
os.environ["REPLICATE_API_TOKEN"] = "r8_GD9vQuJm27to5pD2NQB94N5qr5ihmgj07gWoz"

# Initializing the Llama 2 model using Replicate
llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1}
)

#loading data directly from the excel file
sheet_id1 = '1aR4SCyPBP3ON2yUcLFbVRKp_dQxpjudF6ceTOKLHkIA'
diet_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id1}/export?format=csv')

sheet_id2 = '16f52y1BEI1KWDnS0iUgwDJ2mvmj0-NoxTsig4zBNLcw'
workout_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id2}/export?format=csv')

diet_features = diet_data[['Allergens', 'Vegetarian', 'Dietary Tags', 'Calories']]

# preprocessing the dietary data
diet_data['Allergens'] = diet_data['Allergens'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])
diet_data['Dietary Tags'] = diet_data['Dietary Tags'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])

# reprocessing the workout data
def categorize_activity_level(activity):
    activity = activity.lower().strip()
    if activity in ['sedentary', 'low']:
        return 'Sedentary'
    elif activity in ['light', 'lightly active']:
        return 'Lightly Active'
    elif activity in ['moderate', 'moderately active']:
        return 'Moderately Active'
    elif activity in ['high', 'very active']:
        return 'Very Active'
    elif activity in ['very high', 'extremely active']:
        return 'Extremely Active'
    else:
        return 'Moderately Active'

# applying the categorization from the processed workout data to the activity level
workout_data['Activity Level'] = workout_data['Activity Level'].apply(categorize_activity_level)

# defining the prompt template for diet and workout recommendations
diet_template = PromptTemplate(
    input_variables=['Allergens', 'Vegetarian', 'Dietary_tags'],
    template="""Ghanaian Diet & Workout Recommendation System:
    Allergens : {Allergens}
    Vegetarian: {Vegetarian}
    Dietary_tags: {Dietary_tags}
    """  
)

# defining the prompt template for weight management advice
weight_management_template = PromptTemplate(
    input_variables=["bmi", "bmr", "daily_calories", "diet", "activity_level"],
    template="""Based on the following health metrics and user information, provide 3 to 5 personalized weight management tips in bullet points in one line sentence. don' add anything sentence before and after the points, avoid the sure, .....just list the points. After the title just list the points, no overview. write everything in the second person point of view, instead of user use you:
    BMI: {bmi}
    BMR: {bmr} calories/day
    Estimated Daily Caloric Needs: {daily_calories} calories/day
    Current Diet: {diet}
    Activity Level: {activity_level}
    """
)

# calculating the BMI based on the user's input
def calculating_bmi(weight, height):
    return weight / ((height/100)**2)

# the function to calculate the BMR
def calculating_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

#calculating daily caloric needs
def calorie_needs(bmr, activity_level):
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    return bmr * activity_factors[activity_level]

# function to calculate Body Fat Percentage (BFP)
def calculate_bfp(bmi, age, gender):
    if gender.lower() == 'male':
        return 1.20 * bmi + 0.23 * age - 16.2
    else:
        return 1.20 * bmi + 0.23 * age - 5.4

# function to calculate Body Surface Area (BSA)
def calculate_bsa(weight, height):
    return 0.007184 * (weight ** 0.425) * (height ** 0.725)

# function to calculate Lean Body Mass (LBM)
def calculate_lbm(weight, bfp):
    return weight - (weight * bfp / 100)

# function to get meal recommendations based on the user's preferences
def meal_recommendations(meal_type, user_preferences):
    meal_data = diet_data[diet_data['Meal Type'] == meal_type]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens','Category', 'Vegetarian', 'Calories'])
    
    # filtering the Meal Names based on user preferences
    if user_preferences['vegetarian'] != 'All':
        meal_data = meal_data[meal_data['Vegetarian'] == user_preferences['vegetarian']]
    
    # Filtering out food names with user's allergens
    if user_preferences['allergies'] and 'None' not in user_preferences['allergies']:
        meal_data = meal_data[~meal_data['Allergens'].apply(lambda x: any(allergen in x for allergen in user_preferences['allergies']))]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Category', 'Vegetarian', 'Calories'])
    
    recommendations = meal_data.sample(n=min(5, len(meal_data)))
    return recommendations[['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian', 'Calories']]

# function to get workout recommendations based on activity level and factor
def workout_recommendations(activity_level, activity_factor):
    intensity_map = {
        'Sedentary': 'Low',
        'Lightly Active': 'Low',
        'Moderately Active': 'Medium',
        'Very Active': 'High',
        'Extremely Active': 'High'
    }
    
    intensity = intensity_map[activity_level]
    
    # filtering workouts based on intensity
    filtered_workouts = workout_data[workout_data['Intensity'] == intensity]
    
    if filtered_workouts.empty:
        filtered_workouts = workout_data
    
    # adjusting workout intensity based on activity factor
    if activity_factor <= 1.375:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'].isin(['Low', 'Medium'])]
    elif activity_factor <= 1.725:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'].isin(['Medium', 'High'])]
    else:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'] == 'High']
    
    # getting unique workouts
    unique_workouts = filtered_workouts.drop_duplicates(subset=['Workout Name'])
    
    # Determining the number of recommendations for workouts
    num_unique = len(unique_workouts)
    num_recommendations = min(max(3, num_unique), 7)
    
    # If we have enough unique workouts, sample without replacement
    if num_unique >= num_recommendations:
        recommendations = unique_workouts.sample(n=num_recommendations, replace=False)
    else:
        # If we don't have enough unique workouts, use all unique ones and fill the rest randomly
        recommendations = pd.concat([
            unique_workouts,
            filtered_workouts.sample(n=num_recommendations-num_unique, replace=True)
        ])
    
    # shuffling the recommendations
    recommendations = recommendations.sample(frac=1).reset_index(drop=True)
    
    return recommendations[['Workout Name', 'Duration (minutes)', 'Type of Exercise', 'Reps', 'Sets', 'Distance (km)', 'Intensity']]

# Streamlit UI for the App Title and sub header
st.markdown('<h1 style="color: orange; text-align: center; font-size: 150px;">Didi Yiye</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 20px;">Your Ghanaian Diet & Workout Recommender</h2>', unsafe_allow_html=True)

# accepting User input

with st.expander("Enter Your Information"):
    age = st.number_input('Age', min_value=1, max_value=120)
    weight = st.number_input('Weight (kg)', min_value=1.0, max_value=300.0)
    height = st.number_input('Height (cm)', min_value=1, max_value=300)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    allergies = st.multiselect('Allergies', ['None', 'Fish', 'Nuts', 'Meat', 'Dairy', 'Beans', 'Soy', 'Shellfish', 'Gluten', 'Chicken', 'Beef'])
    vegetarian_option = st.selectbox('Diet Type', ['All', 'Non-vegetarian', 'Vegetarian'])
    vegetarian_mapping = {'All': 'All', 'Vegetarian': 'Yes', 'Non-vegetarian': 'No'}
    vegetarian = vegetarian_mapping[vegetarian_option]
    dietary_tags = st.selectbox('Dietary Tags', ['All', 'High in Carbs', 'High in Protein', 'High in Vegetables', 'High in Protein, Carbs', 'High in Carbs, Protein', 'High in Protein, Fat', 'High in Carbs, Spicy', 'High in Fat, Protein', 'High in Carbs, Fat', 'High in Spice'])
    activity_level = st.selectbox('Activity Level', ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active'])
    
# creating columns for buttons
diet, workout, health = st.columns(3)

#processing the recommendations when buttons are pressed
if diet.button('Diet Recommendations'):
    user_input = {
        "Allergens": allergies,
        "Vegetarian": vegetarian,
        "Dietary_tags": dietary_tags,
        "activity_level": activity_level,
    }
    
    # Generating recommendations
    chain = LLMChain(llm=llm, prompt=diet_template)
    try:
        response = chain.run(user_input)
        
        # Extracting the recommendations using regex
        meal_types = ["Breakfast", "Snack", "Lunch", "Dinner"]
        
        st.subheader("Diet Recommendations")
        
        for meal_type in meal_types:
            st.markdown(f"### {meal_type}")
            
            # Recommendations based on the user preferences
            user_preferences = {
                'vegetarian': vegetarian,
                'allergies': allergies if 'None' not in allergies else []
            }
            recommendations = meal_recommendations(meal_type, user_preferences)
            
            if not recommendations.empty:
                st.table(recommendations)
            else:
                st.write(f"No {meal_type.lower()} recommendations found based on your input.")
    except Exception as e:
        st.error(f"An error occurred while generating diet recommendations: {str(e)}")

if workout.button('Workout Recommendations'):
    st.subheader("Workout Recommendations")
    
    # Calculate activity factor
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    activity_factor = activity_factors[activity_level]
    
    # Workout recommendations based on activity level and factor
    workout_recommendations = workout_recommendations(activity_level, activity_factor)
    
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("No workout recommendations found based on your activity level. You can select another level")

#button to display the calculated health metrics
if health.button('Health Metrics'):
    bmi = calculating_bmi(weight, height)
    bmr = calculating_bmr(weight, height, age, gender)
    daily_calories = calorie_needs(bmr, activity_level)
    bfp = calculate_bfp(bmi, age, gender)
    bsa = calculate_bsa(weight, height)
    lbm = calculate_lbm(weight, bfp)
    
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    
    activity_factor = activity_factors[activity_level]
    
    st.subheader("Health Metrics")
    st.write(f"Body Mass Index (BMI): {bmi:.2f}")
    st.write(f"Basal Metabolic Rate (BMR): {bmr:.2f} calories/day")
    st.write(f"Estimated Daily Caloric Needs: {daily_calories:.2f} calories")
    st.write(f"Body Fat Percentage (BFP): {bfp:.2f}%")
    st.write(f"Body Surface Area (BSA): {bsa:.2f} m²")
    st.write(f"Lean Body Mass (LBM): {lbm:.2f} kg")
    st.write(f"Activity Factor: {activity_factor:.2f}")
    
    # generating weight management tips using Llama 2
    weight_management_chain = LLMChain(llm=llm, prompt=weight_management_template)
    weight_management_input = {
        "bmi": f"{bmi:.2f}",
        "bmr": f"{bmr:.2f}",
        "daily_calories": f"{daily_calories:.2f}",
        "diet": dietary_tags,
        "activity_level": activity_level
    }
    try:
        weight_management_tips = weight_management_chain.run(weight_management_input)
        st.subheader("Weight Management Advice")
        st.write(weight_management_tips)
    except Exception as e:
        st.error(f"An error occurred while generating weight management tips: {str(e)}")
    
    # displaying workout recommendations based on activity factor
    st.subheader("Workouts Based on Health Metrics")
    workout_recommendations = workout_recommendations(activity_level, activity_factor)
    
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("No workout recommendations found based on your activity level. You can select another level")

# running the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("## About")
    st.sidebar.info("This app recommends Ghanaian diet and workout based on your input. It uses a language model to generate recommendations and suggests suitable Food & Workouts based on your input.")
    
    st.sidebar.markdown("## Notice")
    st.sidebar.warning("NB: This app is for informational purposes only and should not replace professional medical advice of any sort")