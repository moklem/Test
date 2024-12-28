pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters,
    and removing stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def calculate_similarity(job_desc, resume_text):
    """
    Calculate similarity between job description and resume using TF-IDF and cosine similarity
    """
    # Preprocess texts
    processed_job_desc = preprocess_text(job_desc)
    processed_resume = preprocess_text(resume_text)
    
    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Combine texts for vectorization
    tfidf_matrix = vectorizer.fit_transform([processed_job_desc, processed_resume])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return cosine_sim

def keyword_match_score(job_desc, resume_text):
    """
    Calculate keyword match score
    """
    job_keywords = set(preprocess_text(job_desc).split())
    resume_keywords = set(preprocess_text(resume_text).split())
    
    matching_keywords = job_keywords.intersection(resume_keywords)
    
    # Calculate keyword match percentage
    keyword_match_percentage = len(matching_keywords) / len(job_keywords) * 100 if job_keywords else 0
    
    return keyword_match_percentage

def rank_applications(df, job_description):
    """
    Rank job applications based on job description
    """
    # Calculate similarity and keyword scores
    df['similarity_score'] = df['resume_text'].apply(lambda x: calculate_similarity(job_description, x))
    df['keyword_score'] = df['resume_text'].apply(lambda x: keyword_match_score(job_description, x))
    
    # Combine scores with weighted average
    df['total_score'] = (0.7 * df['similarity_score'] * 100) + (0.3 * df['keyword_score'])
    
    # Sort by total score in descending order
    return df.sort_values('total_score', ascending=False)

def main():
    st.title('Job Application Ranking Tool')
    
    # Sidebar for job description input
    st.sidebar.header('Job Description')
    job_description = st.sidebar.text_area('Paste Job Description Here')
    
    # File uploader for CSV with job applications
    uploaded_file = st.file_uploader("Upload CSV with Job Applications", type=['csv'])
    
    if uploaded_file is not None:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = ['name', 'resume_text']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {required_columns}")
            return
        
        # Ranking button
        if st.button('Rank Applications'):
            if job_description:
                # Rank applications
                ranked_df = rank_applications(df, job_description)
                
                # Display results
                st.subheader('Ranked Job Applications')
                for index, row in ranked_df.iterrows():
                    st.write(f"**{row['name']}** - Suitability Score: {row['total_score']:.2f}")
                    
                # Optional: Download ranked results
                st.download_button(
                    label="Download Ranked Results",
                    data=ranked_df.to_csv(index=False),
                    file_name='ranked_applications.csv',
                    mime='text/csv'
                )
            else:
                st.warning('Please enter a job description')

if __name__ == '__main__':
    main()
