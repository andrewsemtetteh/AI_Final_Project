import streamlit as st
import pandas as pd
import numpy as np
import os
import replicate
from langchain.prompts import PromptTemplate
from langchain.llms import Replicate
from langchain.chains import LLMChain

# loading our API token
os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_token_here"

#Initializing the Llama 2 model using Replicate
llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1}
)

diet_data = pd.read_csv('diet_data.csv')
workout_data = pd.read_csv('workout_data.csv')

diet_features = diet_data[['Allergens', 'Vegetarian', 'Dietary Tags']]

# preprocessing the dietry data
diet_data['Allergens'] = diet_data['Allergens'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])
diet_data['Dietary Tags'] = diet_data['Dietary Tags'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])

# preprocessing the workout data
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
        return 'Moderately Active'  # default case

#applying the categorization from the processed workout data to the activity level
workout_data['Activity Level'] = workout_data['Activity Level'].apply(categorize_activity_level)

# defining the prompt template for diet and workout recommendations
diet_template = PromptTemplate(
    input_variables=['Allergens', 'Vegetarian', 'Dietary_tags'],
    
    template="""Ghanaian Diet & Workout Recommendation System:
    Based on the following user information, recommend 5 Ghanaian breakfast names, 4 Ghanaian snack names, 5 Ghanaian lunch names and 5 Ghanaian dinner names: 
    Allergens : {Allergens}
    Vegetarian: {Vegetarian}
    Dietary_tags: {Dietary_tags}
    
    Please format your response as follows:
    Breakfast: [List of 5 breakfast names]
    Snack: [List of 4 snack names]
    Lunch: [List of 5 lunch names]
    Dinner: [List of 5 dinner names]
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

    Please provide actionable tips between 3 to 5 for weight management that take into account the user's current situation and health status in bullet points. In one line sentence. Avoid the intro, avoid the sure, .....just list the points.write everything in the second person point of view, instead of user use you:
    """
)

# calculating the BMI based on the users input
def calculating_bmi(weight, height):
    return weight / ((height/100)**2)

# Function to calculate  the BMR
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

# function to get meal recommendations based on the user's preferences
def meal_recommendations(meal_type, user_preferences):
    meal_data = diet_data[diet_data['Meal Type'] == meal_type]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian'])
    
    # filtering the Meal Names based on user preferences
    if user_preferences['vegetarian'] == 'Yes':
        meal_data = meal_data[meal_data['Vegetarian'] == 'Yes']
    elif user_preferences['vegetarian'] == 'No':
        meal_data = meal_data[meal_data['Vegetarian'] == 'No']
    
    # Filtering out food names with user's allergens
    if user_preferences['allergies']:
        meal_data = meal_data[~meal_data['Allergens'].apply(lambda x: any(allergen in x for allergen in user_preferences['allergies']))]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian', 'Calories'])
    
    # Get top 5 recommendations
    recommendations = meal_data.sample(n=min(5, len(meal_data)))
    return recommendations[['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian']]

# Function to get workout recommendations based on activity level
def workout_recommendations(activity_level):
    # mapping the activity levels to workout intensities
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
    
    # Getting unique workouts
    unique_workouts = filtered_workouts.drop_duplicates(subset=['Workout Name'])
    
    # determining the number of recommendations
    num_unique = len(unique_workouts)
    num_recommendations = min(max(3, num_unique), 7)
    
    # if we have enough unique workouts, it should sample without replacement
    if num_unique >= num_recommendations:
        recommendations = unique_workouts.sample(n=num_recommendations, replace=False)
    else:
        # if we don't have enough unique workouts, use all unique ones and fill the rest randomly
        recommendations = pd.concat([
            unique_workouts,
            filtered_workouts.sample(n=num_recommendations-num_unique, replace=True)
        ])
    
    # shuffling the recommendations based on the user input
    recommendations = recommendations.sample(frac=1).reset_index(drop=True)
    
    return recommendations[['Workout Name', 'Duration (minutes)', 'Type of Exercise', 'Intensity']]

# Streamlit UI for the App Title and sub header
st.markdown('<h1 style="color: orange; text-align: center; font-size: 150px;">Didi Yiye</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 20px;">Your Ghanaian Diet & Workout Recommender</h2>', unsafe_allow_html=True)

#accepting User input
age = st.number_input('Age', min_value=1, max_value=120, value=30)
weight = st.number_input('Weight (kg)', min_value=1.0, max_value=300.0, value=70.0)
height = st.number_input('Height (cm)', min_value=1, max_value=300, value=170)
gender = st.selectbox('Gender', ['Male', 'Female'])
allergies = st.multiselect('Allergies', ['None', 'Fish', 'Nuts', 'Meat', 'Dairy', 'Beans', 'Soy', 'Shellfish', 'Gluten', 'Chicken', 'Beef'])
vegetarian_option = st.selectbox('Diet Type', ['Non-vegetarian', 'Vegetarian'])
vegetarian_mapping = {'Vegetarian': 'Yes', 'Non-vegetarian': 'No'}
vegetarian = vegetarian_mapping[vegetarian_option]
dietary_tags = st.selectbox('Dietary Tags',['High in Carbs', 'High in Protein', 'High in Vegetables', 'High in Protein, Carbs', 'High in Carbs, Protein', 'High in Protein, Fat', 'High in Carbs, Spicy', 'High in Carbs, Spicy', 'High in Fat, Protein', 'High in Carbs, Fat', 'High in Spice'])
activity_level = st.selectbox('Activity Level', ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extremely Active'])

# creating columns for buttons
diet, workout, health = st.columns(3)

# processing recommendations when buttons are pressed
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
        
        # extracting the recommendations using regex
        meal_types = ["Breakfast", "Snack", "Lunch", "Dinner"]
        
        st.subheader("Diet Recommendations")
        
        for meal_type in meal_types:
            st.markdown(f"### {meal_type}")
            
            #recommendations based on the user preferences
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
    
    # workout recommendations based on activity level
    workout_recommendations = workout_recommendations(activity_level)
    
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("No workout recommendations found based on your activity level. You can select another level")

# button to display the calculated health metrics
if health.button('Health Metrics'):
    bmi = calculating_bmi(weight, height)
    bmr = calculating_bmr(weight, height, age, gender)
    daily_calories = calorie_needs(bmr, activity_level)
    
    st.subheader("Health Metrics")
    st.write(f"Body Mass Index (BMI): {bmi:.2f}")
    st.write(f"Basal Metabolic Rate (BMR): {bmr:.2f} calories/day")
    st.write(f"Estimated Daily Caloric Needs: {daily_calories:.2f} calories")
    
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
        
# Running the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("## About")
    st.sidebar.info("This app recommends Ghanaian diet and workout based on your input. It uses a language model to generate recommendations and suggests suitable Food & Workouts based on your input.")
    
    st.sidebar.markdown("## Notice")
    st.sidebar.warning("NB: This app is for informational purposes only and should not replace professional medical advice of any sort")