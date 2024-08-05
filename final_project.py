import pandas as pd
import numpy as np
import random
import streamlit as st
from ctransformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import LLMChain

# load the model using ctransformers
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf")

# create a function to generate text using the loaded model
def generate_text(prompt, max_new_tokens=256):
    return model(prompt, max_new_tokens=max_new_tokens)

# loading the diet and workout datasets directly from the google sheets file
sheet_id1 = '1aR4SCyPBP3ON2yUcLFbVRKp_dQxpjudF6ceTOKLHkIA'
diet_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id1}/export?format=csv')

sheet_id2 = '16f52y1BEI1KWDnS0iUgwDJ2mvmj0-NoxTsig4zBNLcw'
workout_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id2}/export?format=csv')

diet_features = diet_data[['Allergens', 'Vegetarian', 'Dietary Tags']]

# preprocessing the diet data
diet_data['Allergens'] = diet_data['Allergens'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])
diet_data['Dietary Tags'] = diet_data['Dietary Tags'].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])

# preprocessing workout data
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

# applying the categorization
workout_data['Activity Level'] = workout_data['Activity Level'].apply(categorize_activity_level)

# defining the prompt template for diet and workout recommendations
diet_template = PromptTemplate(
    input_variables=['Allergens', 'Vegetarian', 'Dietary_tags'],
    template="""Ghanaian Diet & Workout Recommendation System:
    Based on the following user information, recommend between 3 to 5 Ghanaian breakfast names, between 1 to 3 Ghanaian snack names, 
    between 3 to 7 Ghanaian lunch names and between 3 to 5 Ghanaian dinner names: 
    Allergies : {Allergens}
    Vegetarian: {Vegetarian}
    Dietary_tags: {Dietary_tags}
    
    Please format your response as follows:
    Breakfast: [List between 3 to 5 breakfast names]
    Snack: [List between 1 to 3 snack names]
    Lunch: [List between 3 to 7 lunch names]
    Dinner: [List between 3 to 5 dinner names]
    """  
)

# defining the prompt template for weight management tips
weight_management_template = PromptTemplate(
    input_variables=["bmi", "bmr", "daily_calories", "diet", "activity_level"],
    template="""Based on the following health metrics and user information (weight, height, age, BMI, BMI and Estimated Daily Caloric Needs), provide 3 to 5 personalized weight management tips:
    BMI: {bmi}
    BMR: {bmr} calories/day
    Estimated Daily Caloric Needs: {daily_calories} calories/day
    Current Diet: {diet}
    Activity Level: {activity_level}

    Please provide 3 to 5 actionable tips for weight management that take into account the user's current Health (weight, height, age, BMI, BMI and Estimated Daily Caloric Needs).
    """
)

# function to calculate bmi based on the users input
def calculating_bmi(weight, height):
    return weight / ((height/100)**2)

# function to calculate bmr
def calculating_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

# function to calculate daily caloric needs
def calorie_needs(bmr, activity_level):
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    return bmr * activity_factors[activity_level]

# function to get meal recommendations based on user preferences
def meal_recommendations(meal_type, user_preferences):
    meal_data = diet_data[diet_data['Meal Type'] == meal_type]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian', 'Calories'])
    
    # filtering based on vegetarian status
    if user_preferences['vegetarian'] == 'Yes':
        meal_data = meal_data[meal_data['Vegetarian'] == 'Yes']
    elif user_preferences['vegetarian'] == 'No':
        meal_data = meal_data[meal_data['Vegetarian'] == 'No']

    # filtering out food names with user's allergens
    if user_preferences['allergies']:
        meal_data = meal_data[~meal_data['Allergens'].apply(lambda x: any(allergen in x for allergen in user_preferences['allergies']))]
    
    # if no meals left after filtering, an empty dataframe is returned
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian'])
    
    # generates a random number between 3 and 7
    num_recommendations = random.randint(3, 7)

    # getting random recommendations based on the num_recommendations variable
    recommendations = meal_data.sample(n=min(num_recommendations, len(meal_data)))
    return recommendations[['Diet Name', 'Ingredients', 'Allergens', 'Vegetarian']]

# function to get workout recommendations based on activity level
def workout_recommendations(activity_level):
    # mapping the activity levels to workout intensities
    activity_map = {
        'Sedentary': 'Sedentary',
        'Lightly Active': 'Lightly Active',
        'Moderately Active': 'Moderately Active',
        'Very Active': 'Very Active',
        'Extremely Active': 'Extremely Active'
    }
    
    intensity = activity_map[activity_level]
    
    # filtering workouts based on intensity
    filtered_workouts = workout_data[workout_data['Intensity'] == intensity]
    
    # if no workouts match the intensity, use all workouts
    if filtered_workouts.empty:
        filtered_workouts = workout_data
    
    # get unique workouts so none is repeated
    unique_workouts = filtered_workouts.drop_duplicates(subset=['Workout Name'])
    num_unique = len(unique_workouts)
    
    # generate a random number between 5 and 10 for workout recommendations
    num_recommendations = random.randint(5, 10)

    # ensure we do not exceed the number of available unique workouts
    num_recommendations = min(num_recommendations, num_unique)

    # if we have enough unique workouts, sample without replacement
    if num_unique >= num_recommendations:
        recommendations = unique_workouts.sample(n=num_recommendations, replace=False)
    else:
        # if we don't have enough unique workouts, use all unique ones and fill the rest randomly
        recommendations = pd.concat([
            unique_workouts,
            filtered_workouts.sample(n=num_recommendations-num_unique, replace=True)
        ])
    
    # shuffling the recommendations
    recommendations = recommendations.sample(frac=1).reset_index(drop=True)
    
    return recommendations[['Workout Name', 'Duration (minutes)', 'Type of Exercise', 'Intensity']]

# header and title for the application
st.markdown('<h1 style="color: orange; text-align: center; font-size: 150px;">Didi Yiye</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 20px;">The Ultimate Ghanaian Diet & Workout Recommender</h2>', unsafe_allow_html=True)

# accepting user input
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

# creating columns for the diet, workout and health metrics buttons
diet, workout, health = st.columns(3)

# processes recommendations when buttons are pressed
if diet.button('Diet Recommendations'):
    user_input = {
        "Allergens": allergies,
        "Vegetarian": vegetarian,
        "Dietary_tags": dietary_tags,
        "activity_level": activity_level,
    }
    
    # generating recommendations using the loaded model
    prompt = diet_template.format(**user_input)
    response = generate_text(prompt)
    
    st.subheader("Diet Recommendations")
    st.write(response)

if workout.button('Workout Recommendations'):
    st.subheader("Workout Recommendations")
    
    # getting workout recommendations based on activity level
    workout_recommendations = workout_recommendations(activity_level)
    
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("no workout recommendations found based on your activity level. you can select another level")

# button to display the calculated health metrics
if health.button('Health Metrics'):
    bmi = calculating_bmi(weight, height)
    bmr = calculating_bmr(weight, height, age, gender)
    daily_calories = calorie_needs(bmr, activity_level)
    
    st.subheader("Health Metrics")
    st.write(f"BMI: {bmi:.2f}")
    st.write(f"BMR: {bmr:.2f} calories/day")
    st.write(f"Estimated Daily Caloric Needs: {daily_calories:.2f} calories")
    
    # generating weight management tips using the loaded model
    weight_management_input = {
        "bmi": f"{bmi:.2f}",
        "bmr": f"{bmr:.2f}",
        "daily_calories": f"{daily_calories:.2f}",
        "diet": dietary_tags,
        "activity_level": activity_level
    }
    prompt = weight_management_template.format(**weight_management_input)
    weight_management_tips = generate_text(prompt)
    
    st.subheader("Weight Management Advice")
    st.write(weight_management_tips)

# running the streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("## About")
    st.sidebar.info("this app provides ghanaian diet and workout recommendations based on your input. it uses a language model to generate recommendations and suggest suitable options based on your input.")
    
    st.sidebar.markdown("## Notice")
    st.sidebar.warning("nb: this app is for informational purposes only and should not replace professional medical advice of any sought.")