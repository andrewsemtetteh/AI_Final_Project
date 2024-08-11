import streamlit as st
import pandas as pd
import os
import replicate
from langchain.prompts import PromptTemplate
from langchain.llms import Replicate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# loading environment variables from .env file
load_dotenv()
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

# initializing the llama 2 model using replicate
llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1}
)

# loading our dataset directly from the excel file
sheet_id_diet = '1aR4SCyPBP3ON2yUcLFbVRKp_dQxpjudF6ceTOKLHkIA'
diet_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id_diet}/export?format=csv')

sheet_id_workouts = '16f52y1BEI1KWDnS0iUgwDJ2mvmj0-NoxTsig4zBNLcw'
workout_data = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id_workouts}/export?format=csv')

# preprocessing the dietary data, separating the values separated by comma's
diet_data['Allergens'] = diet_data['Allergens'].fillna('').apply(lambda x: [item.strip() for item in str(x).split(',')] if x else [])
diet_data['Dietary Tags'] = diet_data['Dietary Tags'].fillna('').apply(lambda x: [item.strip() for item in str(x).split(',')] if x else [])
diet_data['Meal Type'] = diet_data['Meal Type'].fillna('').apply(lambda x: [item.strip() for item in str(x).split(',')] if x else [])

# reprocessing and categorizing the workout data
def categorizing_activity_level(activity):
    activity = str(activity).lower().strip()
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
workout_data['Activity Level'] = workout_data['Activity Level'].apply(categorizing_activity_level)

# defining the prompt template for the diet recommendations
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
    template="""State what the health metrics calculated mean and how it affects the users health, whether good  or bad in a "bullet point". The summary should be detailed. Use "your" instead of "user". :
    
    And then based on the summary of the health metrics and user information state 3 to 5 weight management tips for the user in "bullet points". Use "your" instead of "user". All should be in bullet points :

    BMI: {bmi}
    BMR: {bmr} calories/day
    Estimated Daily Caloric Needs: {daily_calories} calories/day
    Current Diet: {diet}
    Activity Level: {activity_level}
    """
)

# calculating the bmi based on the user's input
def calculating_bmi(weight, height):
    return weight / ((height/100)**2)

# the function to calculate the bmr
def calculating_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

# calculating daily caloric needs
def calorie_needs(bmr, activity_level):
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    return bmr * activity_factors[activity_level]

# function to calculate body fat percentage (bfp)
def calculate_bfp(bmi, age, gender):
    if gender.lower() == 'male':
        return 1.20 * bmi + 0.23 * age - 16.2
    else:
        return 1.20 * bmi + 0.23 * age - 5.4

# function to calculate body surface area (bsa)
def calculate_bsa(weight, height):
    return 0.007184 * (weight ** 0.425) * (height ** 0.725)

# function to calculate lean body mass (lbm)
def calculate_lbm(weight, bfp):
    return weight - (weight * bfp / 100)

# function to get meal recommendations based on the user's preferences
def meal_recommendations(meal_type, user_preferences):
    meal_data = diet_data[diet_data['Meal Type'].apply(lambda x: meal_type in x)]
    
    if meal_data.empty:
        return pd.DataFrame(columns=['Diet Name', 'Ingredients', 'Allergens', 'Category', 'Vegetarian', 'Calories'])
    
    # filtering the meal names based on user preferences
    if user_preferences['vegetarian'] != 'All':
        meal_data = meal_data[meal_data['Vegetarian'] == user_preferences['vegetarian']]
    
    # filtering out food names with user's allergens
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
    filtered_workouts = workout_data[workout_data['Intensity'] == intensity]
    
    if filtered_workouts.empty:
        filtered_workouts = workout_data
    
    if activity_factor <= 1.375:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'].isin(['Low', 'Medium'])]
    elif activity_factor <= 1.725:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'].isin(['Medium', 'High'])]
    else:
        filtered_workouts = filtered_workouts[filtered_workouts['Intensity'] == 'High']
    
    unique_workouts = filtered_workouts.drop_duplicates(subset=['Workout Name'])
    num_unique = len(unique_workouts)
    num_recommendations = min(max(3, num_unique), 7)
    
    if num_unique >= num_recommendations:
        recommendations = unique_workouts.sample(n=num_recommendations, replace=False)
    else:
        recommendations = pd.concat([
            unique_workouts,
            filtered_workouts.sample(n=num_recommendations-num_unique, replace=True)
        ])
    
    recommendations = recommendations.sample(frac=1).reset_index(drop=True)
    
    return recommendations[['Workout Name', 'Duration (minutes)', 'Type of Exercise', 'Reps', 'Sets', 'Distance (km)', 'Intensity']]

st.markdown('<h1 style="color: orange; text-align: center; font-size: 150px;">Didi Yiye</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 20px;">Your Ghanaian Diet & Workout Recommender</h2>', unsafe_allow_html=True)

# accepting user input
with st.expander("Enter Your Information & Preferences"):
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

# creating columns for the buttons
diet, workout, health = st.columns(3)

# processing the recommendations when buttons are pressed
if diet.button('Diet Recommendations'):
    user_input = {
        "Allergens": allergies,
        "Vegetarian": vegetarian,
        "Dietary_tags": dietary_tags,
        "activity_level": activity_level,
    }
    
    # generating recommendations
    chain = LLMChain(llm=llm, prompt=diet_template)
    response = chain.run(user_input)
    
    meal_types = ["Breakfast", "Snack", "Lunch", "Dinner"]
    
    st.subheader("Diet Recommendations")
    
    for meal_type in meal_types:
        st.markdown(f"### {meal_type}")
        
        # recommendations based on the user preferences
        user_preferences = {
            'vegetarian': vegetarian,
            'allergies': allergies if 'None' not in allergies else []
        }
        recommendations = meal_recommendations(meal_type, user_preferences)
        
        if not recommendations.empty:
            st.table(recommendations)
        else:
            st.write(f"No {meal_type.lower()} recommendations found based on your input.")

if workout.button('Workout Recommendations'):
    st.subheader("Workout Recommendations")
    
    # calculate activity factor
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725,
        'Extremely Active': 1.9
    }
    activity_factor = activity_factors[activity_level]
    
    # workout recommendations based on activity level and factor
    workout_recommendations = workout_recommendations(activity_level, activity_factor)
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("No workout recommendations found based on your activity level. You can select another level")

# button to display the calculated health metrics
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
    
    st.subheader("Your Health Metrics")
    st.write(f"Body Mass Index (BMI): {bmi:.2f}")
    st.write(f"Basal Metabolic Rate (BMR): {bmr:.2f} calories/day")
    st.write(f"Estimated Daily Caloric Needs: {daily_calories:.2f} calories")
    st.write(f"Body Fat Percentage (BFP): {bfp:.2f}%")
    st.write(f"Body Surface Area (BSA): {bsa:.2f} m²")
    st.write(f"Lean Body Mass (LBM): {lbm:.2f} kg")
    st.write(f"Activity Factor: {activity_factor:.2f}")
    
    # generating weight management tips using the llama 2 model
    weight_management_chain = LLMChain(llm=llm, prompt=weight_management_template)
    weight_management_input = {
        "bmi": f"{bmi:.2f}",
        "bmr": f"{bmr:.2f}",
        "daily_calories": f"{daily_calories:.2f}",
        "diet": dietary_tags,
        "activity_level": activity_level
    }
    weight_management_tips = weight_management_chain.run(weight_management_input)
    st.subheader("Weight Management Advice")
    st.write(weight_management_tips)
    
    st.subheader("Recommended Workouts Based on Your Health Metrics")
    workout_recommendations = workout_recommendations(activity_level, activity_factor)
    
    if not workout_recommendations.empty:
        st.table(workout_recommendations.reset_index(drop=True))
    else:
        st.write("No workout recommendations found based on your activity level. You can select another level")

# running the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("## About")
    st.sidebar.info("This app recommends Ghanaian diets and workouts based on your input. It uses a language model to generate recommendations and suggests suitable Food & Workouts based on user input.")
    
    st.sidebar.markdown("## Notice")
    st.sidebar.warning("NB: This app is for informational purposes only and should not replace professional medical advice of any sort.")