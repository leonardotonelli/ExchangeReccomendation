import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from openai import OpenAI
# Define a dictionary to store user responses
user_responses = {}
# Function to load data
def load_data(file_path):
    return pd.read_excel(file_path)
# Function to normalize columns
def normalize_column(column):
    # Ensuring zero division handling
    range = column.max() - column.min()
    if range == 0:
        return column
    else:
        return (column - column.min()) / range
# Function to calculate score (changed by andrea)
def apply_preferences1(df, weights):
    
    # Define a function to calculate the score for a row
    def calculate_score(row):
        score = 0
        for factor, weight in weights.items():
            normalized_column = factor + '_normalized'
            # Ensure the column exists to avoid KeyErrors
            if normalized_column in row:
                score += row[normalized_column] * (weight / 10)  # Normalize weight from 1-10 scale to 0.1-1.0
        return score
    
    # Apply the calculate_score function to each row
    df['score'] = df.apply(calculate_score, axis=1)
    return df
# Function to parse rank
def parse_rank(rank):
    if pd.isna(rank):
        return None
    if isinstance(rank, str) and '-' in rank:
        low, high = map(int, rank.split('-'))
        return (low + high) / 2
    return int(rank)
# Function to apply ranking boost
def apply_ranking_boost(row, ranking_weights, max_rank_boost=0.02):
    boost = 0
    total_weight = sum(ranking_weights.values())
    for rank_type, weight in ranking_weights.items():
        rank_value = row.get(rank_type, None)
        rank_position = parse_rank(rank_value)
        if rank_position is not None:
            normalized_weight = (weight / total_weight) * max_rank_boost
            boost += normalized_weight * (1 / np.log(rank_position + 1))
    return row['score'] + boost
# Function to apply user preferences
def apply_preferences(data, language_choices, language_importance, region_choices, region_importance, climate_choices, climate_importance):
    max_base_boost = 0.02
    language_importance = int(language_importance) if language_importance else 5
    region_importance = int(region_importance) if region_importance else 5
    climate_importance = int(climate_importance) if climate_importance else 5
    def calculate_boosts(choices, importance, max_boost):
        boosts = [max_boost * ((len(choices) - i) / len(choices)) * (importance / 10) for i in range(len(choices))]
        return boosts
    if language_choices:
        data = data[data['Language'].isin(language_choices)]
        language_boosts = calculate_boosts(language_choices, language_importance, max_base_boost)
        for index, language in enumerate(language_choices):
            if index < len(language_boosts):
                data.loc[data['Language'] == language, 'score'] += language_boosts[index]
    if region_choices:
        region_boosts = calculate_boosts(region_choices, region_importance, max_base_boost)
        for index, region in enumerate(region_choices):
            data.loc[data['Region'] == region, 'score'] += region_boosts[index]
    if climate_choices:
        climate_boosts = calculate_boosts(climate_choices, climate_importance, max_base_boost)
        for index, climate in enumerate(climate_choices):
            data.loc[data['Climate'] == climate, 'score'] += climate_boosts[index]
    return data
# Function to only consider available destinations (by course)
def sort_by_courses(df):
    courses = ["CLEAM", "BIEM", "CLEF", "BIEF", "BEMACS", "BAI", "CLEACC", "BEMACC", "BESS", "BIEF-Econ"]
    course_choices = st.multiselect("Choose your course: (you can select multiple)", courses)
    if not course_choices:
        st.write("Please select at least one course.")
        return df
    def course_filter(row):
        if pd.isna(row['Reserved/Not Available']):
            return True  # Keep if no info available
        reserved_info = row['Reserved/Not Available'].strip(" '")
        for course in course_choices:
            if course in row['Reserved/Not Available']:
                if reserved_info.startswith('R'):
                    row['score'] += 0.02
                    return True
                elif reserved_info.startswith('N'):
                    return False
        return reserved_info.startswith('R') == False
    # Apply filter to DataFrame and return result
    filtered_df = df[df.apply(course_filter, axis=1)]
    return filtered_df
                    
# Function to find universities (changed by Andrea)
def find_universities(df):
    try:
        
        user_score = float(st.text_input("Please enter your Exchange score: "))
        available_universities = df[df['Min Score'] <= user_score]
        top_universities = available_universities.sort_values(by='score', ascending=False).head(10)
        if top_universities.empty:
            st.write("No universities available based on your Exchange score.")
        else:
            st.write("Here are the top 10 universities available to you:")
            st.write(top_universities[['University', 'Max Score', 'Min Score']])
    except ValueError:
        st.write("Invalid input. Please enter a valid number for your Exchange score.")
def ask_chatgpt(question, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Instruction for GPT to be concise
    prompt = f"You are a university advisor, at the end of every response ask whether the user as any other question or wants to end typing 'exit', answer coincisely to: {question}"
    data = {
        "model": "gpt-3.5-turbo",  # Ensure you are using the appropriate model
        "messages": [{"role": "user", "content": prompt}]
    }
    print("API Request URL:", url)
    print("API Request Headers:", headers)
    print("API Request Data:", data)
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print("Status Code:", response.status_code)
        print("Response Body:", response.text)
        return "Error fetching response from OpenAI."
def main():
    st.title("University Recommendation Chatbot")
    # Load data
    file_path = 'dataframe_scaped2.xlsx'
    df = load_data(file_path)
    # Normalize columns
    numerical_columns = ['Personal Safety', 'Opportunity to make friends (proportion of youth aged 15-29) ', 'temperature_rating', 'Cost of Living Index']
    for col in numerical_columns:
        df[col + '_normalized'] = normalize_column(df[col])
    # Get user preferences
    st.subheader("User Preferences")
    language_choices = st.multiselect("Choose languages:", df['Language'].unique())
    language_importance = st.slider("How important is language to you?", 1, 10, 5)
    region_choices = st.multiselect("Choose regions:", df['Region'].unique())
    region_importance = st.slider("How important is region to you?", 1, 10, 5)
    climate_choices = st.multiselect("Choose climates:", df['Climate'].unique())
    climate_importance = st.slider("How important is the type of climate to you?", 1, 10, 5)
    #andrea 
    st.subheader("Additional Information")
    
    temperature_importance = st.slider("Temperature: How important is are favorable temperature and humidity to you?", 1, 10, 5)
    cost_of_living_importance = st.slider("Cost Index (the lower, the better): How important is affordability?", 1, 10, 5)
    opp_friends = st.slider("Opportunity of making friends: How important are social opportunities?", 1, 10, 5)
    personal_safety = st.slider("Safety: How important is the safety of the campus and surrounding area?", 1, 10, 5)
    weights = {'temperature_rating': temperature_importance,
               'Cost of Living Index': cost_of_living_importance,
               'Opportunity to make friends (proportion of youth aged 15-29)': opp_friends,
               'Personal Safety': personal_safety}
    df = apply_preferences1(df, weights)
    #
    
    # Apply preferences
    df = apply_preferences(df, language_choices, language_importance, region_choices, region_importance, climate_choices, climate_importance)
    # Get user ranking preferences
    st.subheader("Ranking Preferences")
    ranking_types = df.filter(like='Rankings').columns
    ranking_weights = {}
    for rank_type in ranking_types:
        ranking_weights[rank_type] = st.slider(f"How important is {rank_type}?", 1, 10, 5)

    # Apply ranking boost
    max_ranking_boost = 0.05
    max_ranking_boost = 0.1
    for index, row in df.iterrows():
        df.loc[index, 'score'] = apply_ranking_boost(row, ranking_weights, max_ranking_boost)

    # Find universities based on user's Exchange score
    st.subheader("Find Universities")
    df = sort_by_courses(df)
    client = OpenAI(api_key=st.secrets["API_KEY"])

    find_universities(df)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Chat session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How's NTU reputation worldwide?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("Exchange assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "Exchange assistant", "content": response})


def clear_input():
    st.session_state.chat = ""  # Clear the text input after the message is sent



if __name__ == "__main__":
    main()
