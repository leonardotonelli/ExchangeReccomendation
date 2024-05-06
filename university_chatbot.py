import streamlit as st
import pandas as pd
import numpy as np

# Function to load data
def load_data(file_path):
    return pd.read_excel(file_path)

# Function to normalize columns
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

# Function to calculate score
def calculate_score(row, weights):
    score = 0
    for factor, weight in weights.items():
        normalized_column = factor + '_normalized'
        score += row[normalized_column] * weight
    return score

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

# Function to find universities
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

def main():
    st.title("University Recommendation Chatbot")

    # Load data
    file_path = 'dataframe_scraped2.xlsx'
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
    climate_importance = st.slider("How important is climate to you?", 1, 10, 5)

    # Apply preferences
    df = apply_preferences(df, language_choices, language_importance, region_choices, region_importance, climate_choices, climate_importance)

    # Get user ranking preferences
    st.subheader("Ranking Preferences")
    ranking_types = df.filter(like='Rankings').columns
    ranking_weights = {}
    for rank_type in ranking_types:
        ranking_weights[rank_type] = st.slider(f"How important is {rank_type}?", 1, 10, 5)

    # Apply ranking boost
    max_ranking_boost = 0.02
    for index, row in df.iterrows():
        df.loc[index, 'score'] = apply_ranking_boost(row, ranking_weights, max_ranking_boost)

    # Find universities based on user's Exchange score
    st.subheader("Find Universities")
    find_universities(df)

if __name__ == "__main__":
    main()