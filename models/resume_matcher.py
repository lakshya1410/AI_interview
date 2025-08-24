import os
import re
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List, Tuple

load_dotenv()

class ResumeMatcher:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def analyze_resume(self, resume_path: str, job_description: str) -> Dict:
        """Analyze resume against job description using Gemini API"""
        # Extract and preprocess resume text
        resume_text = self.extract_text_from_pdf(resume_path)
        resume_text = self.preprocess_text(resume_text)
        job_description = self.preprocess_text(job_description)
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze the following resume against the job description and provide a detailed assessment:
        
        RESUME:
        {resume_text}
        
        JOB DESCRIPTION:
        {job_description}
        
        Please provide your analysis in the following JSON format:
        {{
            "match_score": <score from 0-100>,
            "strengths": [
                "<strength1>",
                "<strength2>",
                "<strength3>"
            ],
            "weaknesses": [
                "<weakness1>",
                "<weakness2>",
                "<weakness3>"
            ],
            "missing_skills": [
                "<missing_skill1>",
                "<missing_skill2>"
            ],
            "recommendations": [
                "<recommendation1>",
                "<recommendation2>"
            ],
            "summary": "<brief overall assessment>"
        }}
        
        Consider technical skills, experience relevance, education, and overall fit for the role.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_gemini_response(response.text)
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "match_score": 0,
                "strengths": [],
                "weaknesses": [],
                "missing_skills": [],
                "recommendations": [],
                "summary": "Analysis could not be completed"
            }
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini API response and extract JSON"""
        try:
            # Extract JSON from response (handle potential markdown formatting)
            import json
            
            # Find JSON content between ```json and ``` or just raw JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                json_str = json_match.group(0) if json_match else response_text
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: parse manually or return structured error
            return {
                "match_score": 50,
                "strengths": ["Unable to parse detailed analysis"],
                "weaknesses": ["Analysis format error"],
                "missing_skills": [],
                "recommendations": ["Please try again"],
                "summary": response_text[:500] + "..." if len(response_text) > 500 else response_text
            }
