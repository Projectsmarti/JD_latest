import streamlit as st
import pandas as pd
import os
import re
import ast
import random
import google.generativeai as palm
from langchain.llms import GooglePalm
from fuzzywuzzy import fuzz

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)

def hybrid_similarity(jd_skills, resume_skills, threshold):
    try:
        set1 = set(jd_skills)
        set2 = set(resume_skills)

        matched_skills = list(set1.intersection(set2))  # Keep track of matched skills
        not_matched = set1 - set2
        intersection_size = len(matched_skills)  # Initialize intersection_size here

        if not_matched:
            mtchd_after_lvn = []
            for nm_skills in not_matched:
                for rsm_skills in resume_skills:
                    leven_metric = fuzz.ratio(nm_skills, rsm_skills)
                    if leven_metric >= threshold:
                        mtchd_after_lvn.append(nm_skills)

            matched_skills.extend(mtchd_after_lvn)  # Add these to matched skills
            intersection_size += len(mtchd_after_lvn)
            left_over_skills = set(not_matched) - set(mtchd_after_lvn)

            if left_over_skills:
                final_skl_mtch = []
                for skl in left_over_skills:
                    for rsm_skl in resume_skills:
                        if skl in rsm_skl:
                            final_skl_mtch.append(skl)

                matched_skills.extend(final_skl_mtch)  # Add these to matched skills
                intersection_size += len(final_skl_mtch)

        union_size = len(set1)
        if union_size == 0:
            return 0, []

        return intersection_size / union_size, matched_skills
    except Exception as e:
        print(f"An error occurred in the hybrid_similarity function: {str(e)}")
        return None, None

def extract_between_chars_regex(input_string, start_char, end_char):
    try:
        pattern = re.compile(f'{re.escape(start_char)}(.*?){re.escape(end_char)}')
        match = pattern.search(input_string)

        if match:
            return match.group(1)
        else:
            return None
    except Exception as e:
        print(f"An error occurred in the extract_between_chars_regex function: {str(e)}")
        return None

def jd_skills_data_prep(text):
    try:
        skills = str(text).lower()
        skills = extract_between_chars_regex(skills, '[', ']')
        if skills is not None:
            skills = skills.replace('"', '').replace("'", "").replace(")", "").replace(" and", ", ").replace("&",
                                                                                                                ", ")
            skills = skills.split(", ")
        else:
            skills = []
        return skills
    except Exception as e:
        print(f"An error occurred in the jd_skills_data_prep function: {str(e)}")
        return []

def get_palm_response(text, prompt):
    try:
        os.environ['GOOGLE_API_KEY'] = 'AIzaSyCmdhOVj_KcpTxpWXH94DJOnBuXfZGZffg'
        palm.configure(api_key=os.environ['GOOGLE_API_KEY'])
        llm = GooglePalm()
        llm.temperature = 0.1
        llm_result = llm._generate([text + prompt])

        return llm_result.generations[0][0].text
    except Exception as e:
        print(f"An error occurred in the get_palm_response function: {str(e)}")
        return None

def get_jd_skills_and_exp(jd_text):
    # Prompts
    prompt1 = " Return python list with skill names only picked from above text"
    prompt2 = " Return minimum experience in years number only"

    # Skills
    skills = get_palm_response(prompt1, jd_text)
    skills = skills.lower()
    try:
        skills = ast.literal_eval(skills)
    except:
        skills = jd_skills_data_prep(skills)

    # Experience
    try:
        experience = float(get_palm_response(prompt2, jd_text))
    except Exception as e:
        print(f"An error occurred while converting experience to float: {str(e)}")
        experience = None

    return jd_text, skills, experience

st.set_page_config(
    page_title="JD Parsing App",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a Bug': 'https://www.extremelycoolapp.com/bug',
        'About': 'This is a header. This is an extremely cool app!'
    }
)

st.markdown("<h1 style='text-align: center; color: Blue'>JD & RESUME MATCHING MATRIX </h1>",
            unsafe_allow_html=True)

st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Select an Option", ["Extract JD"])

jd_skills = ""
jd_experience = ""
jd_full_text = ""
if selected_option == "Upload File":
    st.title('JD File')

    uploaded_file = st.file_uploader("Choose a job description file", type=['txt', 'csv', 'docx', 'pdf'])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.markdown("<h2 style='text-align: center; color: #3498db;'>Job Description</h2>",
                    unsafe_allow_html=True)
        st.table(data[['Text']])

else:
    st.markdown("<h3 style='text-align: left; color: Red'>Paste your JD Here </h3>",
                unsafe_allow_html=True)

    jd_full_text = st.text_area('', height=200)
    st.markdown(
        """
        <style>
        .stButton > button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """, unsafe_allow_html=True)

    if jd_full_text.strip() == '':
        st.warning("No JD passed.")
    elif len(jd_full_text.split()) < 25:
        st.warning("Input has less than 25 words.")
    else:
        if st.button("Extract Skills and Experience"):
            jd_full_text, jd_skills, jd_experience = get_jd_skills_and_exp(jd_full_text)
            st.write(f"SKILLS REQUIRED: {jd_skills}")
            st.write(f"EXPERIENCE REQUIRED: {jd_experience}")

        resume_data = pd.read_csv("Resume_Parsed_Sample_v4_with_exp_refurb.csv")

        if st.button("Matched Resumes"):
            jd_full_text, jd_skills, jd_experience = get_jd_skills_and_exp(jd_full_text)
            st.write(f"SKILLS REQUIRED: {jd_skills}")
            st.write(f"EXPERIENCE REQUIRED: {jd_experience}")

            threshold = 90
            final_list = []
            for j, res_row in resume_data.iterrows():
                jd_skill_similarity, matched_skills = hybrid_similarity(jd_skills, eval(res_row[3]), threshold)
                Missing_Skills = list(set(jd_skills) - set(eval(res_row[3])))
                additional_skills = list(set(eval(res_row[3])) - set(jd_skills))
                # matched_skills = list(set(matched_skills))
                # Calculate matched skills
                matched_skills = list(set(jd_skills) - set(Missing_Skills))

                final_list.append(
                    [jd_skills, jd_experience, res_row[0], res_row[3], additional_skills, res_row[5],
                     jd_skill_similarity, matched_skills])

            final_data = pd.DataFrame(final_list,
                                      columns=['JD_Skills', 'JD_Experience', 'Sl.No', 'Required_Skills',
                                               'Additional_skills', 'Experience', 'Skill_Similarity',
                                               'Matched_Skills'])

            df_xlsx = pd.read_csv('unicode-update_ltst_resume.csv')
            df_xlsx.rename(columns={'resume_index': 'Sl.No'}, inplace=True)

            final_data = pd.merge(final_data,
                                  df_xlsx[['Sl.No', 'Unique_ID', 'Name', 'Phone Number', 'Email id',
                                           'Location of work', 'Position Applied For']], on='Sl.No')

            final_data['Experience_Tag'] = final_data[['JD_Experience', 'Experience']].apply(
                lambda x: 1 if x['Experience'] >= x['JD_Experience'] else 0, axis=1)

            final_data['Matching_Score'] = final_data[['Skill_Similarity', 'Experience_Tag']].apply(
                lambda x: (x['Skill_Similarity'] + x['Experience_Tag']) / 2 if x['Skill_Similarity'] > 0 else 0,
                axis=1)
            final_data['Additional_skills'] = final_data['Additional_skills'].apply(
                lambda x: 'No additional skills' if not x else x)
            # final_data['Matched_Skills'] = final_data['Matched_Skills'].apply(lambda x: x if x else 'No skills matched')

            final_data = final_data.sort_values(['Matching_Score'], ascending=[False]).reset_index(drop=True)
            final_data['Matching_Score'] = final_data['Matching_Score'].apply(
                lambda x: str(int(x * 100)) + '%')
            top_5_matches = final_data[['Unique_ID', 'Name', 'Matching_Score', 'Experience', 'Matched_Skills',
                                        'Additional_skills', 'Phone Number', 'Email id']]
            top_5_matches = top_5_matches.head(5)
            top_5_matches

