import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Dict, List

load_dotenv()

class QuestionGenerator:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_interview_questions(self, analysis_results: Dict, resume_text: str, job_description: str) -> Dict:
        """Generate structured interview questions based on resume analysis"""
        
        # Create comprehensive prompt for question generation
        prompt = f"""
        Based on the resume analysis and job requirements, generate a comprehensive 20-minute interview question set.
        
        RESUME ANALYSIS:
        - Match Score: {analysis_results.get('match_score', 0)}%
        - Strengths: {', '.join(analysis_results.get('strengths', []))}
        - Weaknesses: {', '.join(analysis_results.get('weaknesses', []))}
        - Missing Skills: {', '.join(analysis_results.get('missing_skills', []))}
        
        RESUME CONTENT:
        {resume_text}
        
        JOB DESCRIPTION:
        {job_description}
        
        Generate interview questions in the following JSON format for a 20-minute interview (15-20 questions):
        {{
            "interview_structure": {{
                "total_duration": "20 minutes",
                "question_count": <number>,
                "difficulty_distribution": {{
                    "easy": <count>,
                    "medium": <count>,
                    "hard": <count>
                }}
            }},
            "question_categories": {{
                "introduction": {{
                    "duration": "2-3 minutes",
                    "questions": [
                        {{
                            "id": 1,
                            "question": "<question>",
                            "type": "introduction",
                            "difficulty": "easy",
                            "time_allocation": "1-2 minutes",
                            "purpose": "<why this question>",
                            "expected_topics": ["<topic1>", "<topic2>"]
                        }}
                    ]
                }},
                "technical_skills": {{
                    "duration": "8-10 minutes",
                    "questions": [
                        {{
                            "id": 2,
                            "question": "<specific technical question based on resume>",
                            "type": "technical",
                            "difficulty": "medium",
                            "time_allocation": "2-3 minutes",
                            "purpose": "<assessment focus>",
                            "expected_topics": ["<topic1>", "<topic2>"],
                            "follow_up": "<potential follow-up question>"
                        }}
                    ]
                }},
                "experience_based": {{
                    "duration": "5-6 minutes",
                    "questions": [
                        {{
                            "id": <id>,
                            "question": "<question about specific experience from resume>",
                            "type": "behavioral",
                            "difficulty": "medium",
                            "time_allocation": "2-3 minutes",
                            "purpose": "<what to assess>",
                            "expected_topics": ["<topic1>", "<topic2>"]
                        }}
                    ]
                }},
                "problem_solving": {{
                    "duration": "3-4 minutes",
                    "questions": [
                        {{
                            "id": <id>,
                            "question": "<scenario-based question>",
                            "type": "problem_solving",
                            "difficulty": "hard",
                            "time_allocation": "3-4 minutes",
                            "purpose": "<assessment goal>",
                            "expected_topics": ["<topic1>", "<topic2>"]
                        }}
                    ]
                }},
                "closing": {{
                    "duration": "1-2 minutes",
                    "questions": [
                        {{
                            "id": <id>,
                            "question": "<closing question>",
                            "type": "closing",
                            "difficulty": "easy",
                            "time_allocation": "1-2 minutes",
                            "purpose": "wrap up and candidate questions",
                            "expected_topics": ["questions", "clarifications"]
                        }}
                    ]
                }}
            }},
            "assessment_focus": [
                "<key skill 1 to assess>",
                "<key skill 2 to assess>",
                "<key skill 3 to assess>"
            ],
            "red_flags_to_watch": [
                "<potential concern 1>",
                "<potential concern 2>"
            ],
            "interview_tips": [
                "<tip for interviewer 1>",
                "<tip for interviewer 2>"
            ]
        }}
        
        Requirements:
        1. Questions should be strictly based on the candidate's resume and job requirements
        2. Progressive difficulty - start easy, build complexity
        3. Focus on areas where candidate shows strength and probe weaknesses
        4. Include specific technical questions related to technologies mentioned in resume
        5. Address missing skills through hypothetical scenarios
        6. Total 15-20 questions for 20-minute duration
        7. Each question should have clear time allocation and assessment purpose
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_questions_response(response.text)
        except Exception as e:
            return self._generate_fallback_questions(analysis_results, job_description)
    
    def _parse_questions_response(self, response_text: str) -> Dict:
        """Parse Gemini API response and extract structured questions"""
        try:
            import re
            
            # Find JSON content
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                json_str = json_match.group(0) if json_match else response_text
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return self._generate_fallback_questions({}, "")
    
    def _generate_fallback_questions(self, analysis_results: Dict, job_description: str) -> Dict:
        """Generate basic questions if API parsing fails"""
        return {
            "interview_structure": {
                "total_duration": "20 minutes",
                "question_count": 15,
                "difficulty_distribution": {"easy": 5, "medium": 7, "hard": 3}
            },
            "question_categories": {
                "introduction": {
                    "duration": "2-3 minutes",
                    "questions": [
                        {
                            "id": 1,
                            "question": "Can you walk me through your background and what brings you to this role?",
                            "type": "introduction",
                            "difficulty": "easy",
                            "time_allocation": "2-3 minutes",
                            "purpose": "Break the ice and understand candidate motivation",
                            "expected_topics": ["background", "motivation", "career goals"]
                        }
                    ]
                },
                "technical_skills": {
                    "duration": "8-10 minutes",
                    "questions": [
                        {
                            "id": 2,
                            "question": "Tell me about a challenging technical project you've worked on recently.",
                            "type": "technical",
                            "difficulty": "medium",
                            "time_allocation": "3-4 minutes",
                            "purpose": "Assess technical depth and problem-solving",
                            "expected_topics": ["technical challenges", "solutions", "technologies"],
                            "follow_up": "What would you do differently if you had to start over?"
                        }
                    ]
                },
                "experience_based": {
                    "duration": "5-6 minutes",
                    "questions": [
                        {
                            "id": 3,
                            "question": "Describe a time when you had to learn a new technology quickly for a project.",
                            "type": "behavioral",
                            "difficulty": "medium",
                            "time_allocation": "2-3 minutes",
                            "purpose": "Assess learning ability and adaptability",
                            "expected_topics": ["learning approach", "challenges", "outcomes"]
                        }
                    ]
                },
                "problem_solving": {
                    "duration": "3-4 minutes",
                    "questions": [
                        {
                            "id": 4,
                            "question": "How would you approach debugging a performance issue in a production system?",
                            "type": "problem_solving",
                            "difficulty": "hard",
                            "time_allocation": "3-4 minutes",
                            "purpose": "Evaluate systematic thinking and troubleshooting skills",
                            "expected_topics": ["debugging methodology", "tools", "systematic approach"]
                        }
                    ]
                },
                "closing": {
                    "duration": "1-2 minutes",
                    "questions": [
                        {
                            "id": 5,
                            "question": "Do you have any questions about the role or our team?",
                            "type": "closing",
                            "difficulty": "easy",
                            "time_allocation": "1-2 minutes",
                            "purpose": "Allow candidate to ask questions and wrap up",
                            "expected_topics": ["role clarification", "team dynamics", "next steps"]
                        }
                    ]
                }
            },
            "assessment_focus": ["Technical competency", "Problem-solving ability", "Communication skills"],
            "red_flags_to_watch": ["Inconsistent explanations", "Lack of technical depth"],
            "interview_tips": ["Listen for specific examples", "Probe for technical details", "Assess cultural fit"]
        }
    
    def get_question_sequence(self, questions_data: Dict) -> List[Dict]:
        """Extract questions in sequential order for interview flow"""
        sequence = []
        question_id = 1
        
        categories = questions_data.get('question_categories', {})
        category_order = ['introduction', 'technical_skills', 'experience_based', 'problem_solving', 'closing']
        
        for category in category_order:
            if category in categories:
                for question in categories[category].get('questions', []):
                    question['category'] = category
                    question['sequence_id'] = question_id
                    sequence.append(question)
                    question_id += 1
        
        return sequence
    
    def get_interview_summary(self, questions_data: Dict) -> Dict:
        """Get interview overview and preparation summary"""
        return {
            "structure": questions_data.get('interview_structure', {}),
            "focus_areas": questions_data.get('assessment_focus', []),
            "red_flags": questions_data.get('red_flags_to_watch', []),
            "tips": questions_data.get('interview_tips', []),
            "total_questions": len(self.get_question_sequence(questions_data))
        }
