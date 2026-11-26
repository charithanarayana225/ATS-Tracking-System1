
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

# ==========================================
# 2. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ATS Resume Expert",
    page_icon="📄",
    layout="wide"
)

# ==========================================
# 3. GEMINI API CONFIGURATION
# ==========================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key is missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# Models confirmed in your model-list output.
# Try lightweight models first.

MODEL_NAMES = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]

# ==========================================
# 4. GEMINI RESPONSE FUNCTION
# ==========================================

def get_gemini_response(prompt, job_description, pdf_bytes):

    pdf_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type="application/pdf"
    )

    contents = [
        prompt,
        f"Job Description:\n{job_description}",
        pdf_part
    ]

    last_error = None

    for model_name in MODEL_NAMES:

        try:
            st.info(f"Connecting to Gemini: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048
                )
            )

            if response.text:
                return response.text

            raise ValueError(
                f"{model_name} returned an empty response."
            )

        except errors.APIError as e:

            last_error = e

            if e.code in (503, 404):

                st.warning(
                    f"{model_name} is unavailable. "
                    "Trying the next model..."
                )

                continue

            # Do not retry authentication or quota errors
            raise

    if last_error:
        raise last_error

    raise RuntimeError("No Gemini model is available.")

# ==========================================
# 5. PROMPT 1 - RESUME EVALUATION
# ==========================================

input_prompt1 = """
You are an experienced Technical Human Resource Manager.

Review the provided resume against the job description.

Provide a professional evaluation covering:

1. Candidate Profile Summary
2. Relevant Technical Skills
3. Work Experience
4. Strengths
5. Weaknesses
6. Educational Qualifications
7. Overall Suitability for the Role

Provide a detailed professional evaluation.

Do not invent qualifications, skills, or experience.
Only use information available in the resume.
"""

# ==========================================
# 6. PROMPT 2 - SKILLS IMPROVEMENT
# ==========================================

input_prompt2 = """
You are an experienced Technical HR Manager.

Analyze the candidate's resume against the job description.

Provide recommendations covering:

1. Skills the Candidate Already Possesses
2. Missing Technical Skills
3. Skills That Need Improvement
4. Recommended Technologies to Learn
5. Certifications That Could Help
6. Practical Learning Roadmap
7. Resume Improvement Suggestions

Provide actionable recommendations.

Prioritize skills based on the job description.
Do not invent qualifications or experience.
"""

# ==========================================
# 7. PROMPT 3 - ATS PERCENTAGE MATCH
# ==========================================

input_prompt3 = """
You are an expert Applicant Tracking System (ATS) scanner.

Evaluate the resume against the provided job description.

Provide your response in the following format:

1. Overall Resume Match Percentage
2. Matching Keywords
3. Missing Keywords
4. Technical Skills Match
5. Experience Match
6. Educational Qualification Match
7. Areas for Improvement
8. Final Recommendations

Give a realistic match percentage based on the resume
content and job requirements.

Do not invent qualifications or experience.

Explain briefly how the match percentage was assessed.
"""

# ==========================================
# 8. APPLICATION HEADER
# ==========================================

st.title("📄 ATS Tracking System")

st.subheader("ATS Resume Expert")

st.write(
    "Upload your resume and compare it with a job "
    "description using Google's Gemini AI."
)

st.divider()

# ==========================================
# 9. JOB DESCRIPTION INPUT
# ==========================================

input_text = st.text_area(
    "Enter Job Description:",
    height=200,
    placeholder="Paste the job description here..."
)

# ==========================================
# 10. RESUME PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your resume (PDF only):",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(
        f"Resume uploaded successfully: {uploaded_file.name}"
    )

# ==========================================
# 11. ACTION BUTTONS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    submit1 = st.button(
        "Tell Me About the Resume",
        use_container_width=True
    )

with col2:

    submit2 = st.button(
        "How Can I Improve My Skills?",
        use_container_width=True
    )

with col3:

    submit3 = st.button(
        "Percentage Match",
        use_container_width=True
    )

# ==========================================
# 12. PROCESS RESUME
# ==========================================

if submit1 or submit2 or submit3:

    if uploaded_file is None:

        st.warning(
            "Please upload your resume in PDF format."
        )

    elif not input_text.strip():

        st.warning(
            "Please enter the job description."
        )

    else:

        pdf_bytes = uploaded_file.getvalue()

        # Validate PDF file
        if not pdf_bytes.startswith(b"%PDF"):

            st.error(
                "Invalid PDF file. Please upload a valid PDF."
            )

        else:

            # Select appropriate prompt
            if submit1:

                selected_prompt = input_prompt1
                heading = "Resume Evaluation"

            elif submit2:

                selected_prompt = input_prompt2
                heading = "Skills Improvement Recommendations"

            else:

                selected_prompt = input_prompt3
                heading = "ATS Percentage Match"

            # ======================================
            # 13. GENERATE GEMINI RESPONSE
            # ======================================

            try:

                with st.spinner(
                    "Analyzing your resume using Gemini AI..."
                ):

                    response = get_gemini_response(
                        selected_prompt,
                        input_text,
                        pdf_bytes
                    )

                st.divider()

                st.subheader(heading)

                st.markdown(response)

                st.success(
                    "Resume analysis completed successfully!"
                )

            # ======================================
            # 14. API ERROR HANDLING
            # ======================================

            except errors.APIError as e:

                error_code = e.code

                if error_code == 503:

                    st.error(
                        "All configured Gemini models are "
                        "currently overloaded. Please try again later."
                    )

                elif error_code == 429:

                    st.error(
                        "Gemini API quota or rate limit exceeded. "
                        "Check your Google AI Studio API usage."
                    )

                elif error_code == 400:

                    st.error(
                        f"Invalid API request: {str(e)}"
                    )

                elif error_code == 403:

                    st.error(
                        "API access denied. Check your API key "
                        "and project permissions."
                    )

                elif error_code == 404:

                    st.error(
                        "The requested Gemini model was not found. "
                        "Check model availability."
                    )

                else:

                    st.error(
                        f"Gemini API Error ({error_code}): {str(e)}"
                    )

            # ======================================
            # 15. GENERAL ERROR HANDLING
            # ======================================

            except Exception as e:

                st.error(
                    f"Unexpected error occurred: {str(e)}"
                )