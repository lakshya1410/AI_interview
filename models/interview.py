# models/enhanced_interview.py
import os
import json
import time
import base64
import sqlite3
import tempfile
import wave
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class EnhancedInterviewSimulator:
    def __init__(self):
        # Configure APIs
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize Groq client for STT
        self.groq_client = None
        self.stt_available = False
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                self.stt_available = True
                print("✅ Groq STT initialized successfully")
        except Exception as e:
            print(f"❌ Groq STT initialization failed: {e}")
        
        # Initialize TTS using Hugging Face
        self.tts_available = False
        try:
            import requests
            self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if self.hf_api_key:
                self.tts_available = True
                print("✅ Hugging Face TTS available")
            else:
                print("⚠️ Hugging Face API key not found, will use browser TTS")
        except Exception as e:
            print(f"⚠️ TTS initialization failed, will use browser TTS: {e}")
        
        # Initialize database
        self.db_path = Path("data/interviews.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Interview state
        self.current_interview = None
        self.conversation_history = []
        self.current_question_index = 0
        self.start_time = None
        self.questions = []
        self.audio_enabled = True
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interviews (
                    id TEXT PRIMARY KEY,
                    candidate_name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    resume_analysis TEXT,
                    questions TEXT,
                    conversation_history TEXT,
                    final_report TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database initialized")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    def text_to_speech_hf(self, text: str) -> Dict:
        """Generate speech using Hugging Face API with fallback"""
        # Always provide browser TTS as fallback
        fallback_response = {
            "success": True,
            "text": text,
            "use_browser_tts": True
        }
        
        if not self.tts_available:
            print("TTS: Using browser fallback (no API key)")
            return fallback_response
        
        try:
            import requests
            
            API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": text[:500]},  # Limit text length
                timeout=10
            )
            
            if response.status_code == 200:
                audio_data = base64.b64encode(response.content).decode('utf-8')
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "format": "wav",
                    "text": text
                }
            else:
                print(f"TTS: HF API error {response.status_code}, using browser fallback")
                return fallback_response
                
        except Exception as e:
            print(f"TTS error: {e}, using browser fallback")
            return fallback_response
    
    def transcribe_audio_enhanced(self, audio_file_path: str) -> Dict:
        """Enhanced audio transcription with format conversion"""
        if not self.stt_available:
            print("STT: Groq client not available")
            return {
                "success": False,
                "error": "Speech-to-text not available. Please type your response.",
                "fallback_to_text": True
            }
        
        try:
            # Validate file
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError("Audio file not found")
            
            file_size = os.path.getsize(audio_file_path)
            print(f"STT: Processing audio file: {audio_file_path}, size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            # Convert WebM to WAV if needed
            converted_path = self._convert_audio_format(audio_file_path)
            print(f"STT: Using audio file: {converted_path}")
            
            with open(converted_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
                
                # Extract text from response
                if hasattr(transcription, 'text'):
                    text = transcription.text.strip()
                elif hasattr(transcription, 'transcription'):
                    text = transcription.transcription.strip()
                else:
                    text = str(transcription).strip()
                
                print(f"STT: Transcription result: '{text}'")
                
                # Clean up converted file
                if converted_path != audio_file_path and os.path.exists(converted_path):
                    try:
                        os.remove(converted_path)
                    except Exception as e:
                        print(f"Warning: Could not remove temp file: {e}")
                
                if not text:
                    return {
                        "success": False,
                        "error": "No speech detected in audio. Please try speaking louder or closer to the microphone.",
                        "fallback_to_text": True
                    }
                
                return {
                    "success": True,
                    "transcription": text,
                    "confidence": getattr(transcription, 'confidence', 0.9)
                }
                
        except Exception as e:
            print(f"STT: Transcription error: {str(e)}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}. Please type your response instead.",
                "fallback_to_text": True
            }
    
    def _convert_audio_format(self, audio_file_path: str) -> str:
        """Convert audio to WAV format for better compatibility"""
        file_ext = Path(audio_file_path).suffix.lower()
        
        # If already WAV, return as is
        if file_ext == '.wav':
            return audio_file_path
        
        try:
            output_path = audio_file_path.replace(file_ext, '.wav')
            
            # Try using ffmpeg-python if available
            try:
                import ffmpeg
                print(f"Converting {audio_file_path} to {output_path} using ffmpeg")
                (
                    ffmpeg
                    .input(audio_file_path)
                    .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
                    .overwrite_output()
                    .run(quiet=True)
                )
                print("Audio conversion successful")
                return output_path
            except ImportError:
                print("ffmpeg not available, trying direct file processing")
                # Try to process the original file directly
                # Many audio processing libraries can handle WebM
                return audio_file_path
                
        except Exception as e:
            print(f"Audio conversion failed: {e}, using original file")
            return audio_file_path
    
    def start_interview(self, questions_data: Dict, candidate_name: str, resume_analysis: Dict) -> Dict:
        """Start interview with enhanced error handling"""
        try:
            self.current_interview = {
                "id": f"interview_{int(time.time())}",
                "candidate_name": candidate_name,
                "start_time": datetime.now(),
                "resume_analysis": resume_analysis,
                "questions": self._extract_questions(questions_data),
                "status": "active"
            }
            
            self.conversation_history = []
            self.current_question_index = 0
            self.questions = self.current_interview["questions"]
            
            # Save to database
            self._save_interview_state()
            
            # Generate personalized opening
            opening_message = self._generate_opening_message(candidate_name)
            
            # Generate TTS
            tts_result = self.text_to_speech_hf(opening_message)
            
            return {
                "success": True,
                "interview_id": self.current_interview["id"],
                "status": "started",
                "ai_message": opening_message,
                "ai_audio": tts_result,
                "total_questions": len(self.questions),
                "video_visible": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start interview: {str(e)}"
            }
    
    def _extract_questions(self, questions_data: Dict) -> List[Dict]:
        """Extract questions in proper sequence"""
        questions = []
        question_id = 1
        
        if questions_data.get('question_categories'):
            category_order = ['introduction', 'technical_skills', 'experience_based', 'problem_solving', 'closing']
            for category in category_order:
                if category in questions_data['question_categories']:
                    for q in questions_data['question_categories'][category]['questions']:
                        questions.append({
                            "id": question_id,
                            "category": category,
                            "question": q['question'],
                            "difficulty": q.get('difficulty', 'medium'),
                            "time_allocation": q.get('time_allocation', '2-3 minutes')
                        })
                        question_id += 1
        
        return questions
    
    def _generate_opening_message(self, candidate_name: str) -> str:
        """Generate personalized opening message"""
        return f"Hello {candidate_name}! Welcome to your AI interview session. I'm your virtual interviewer today. We'll be going through {len(self.questions)} carefully crafted questions that will help assess your skills and experience. Please feel comfortable, speak naturally, and take your time with each response. Are you ready to begin with our first question?"
    
    def get_next_question(self) -> Dict:
        """Get the next interview question"""
        try:
            if self.current_question_index >= len(self.questions):
                completion_message = "Congratulations! You've completed all the interview questions. Let me analyze your responses and generate a comprehensive report for you."
                tts_result = self.text_to_speech_hf(completion_message)
                
                return {
                    "success": True,
                    "status": "completed",
                    "ai_message": completion_message,
                    "ai_audio": tts_result,
                    "show_report": True
                }
            
            current_q = self.questions[self.current_question_index]
            ai_message = self._generate_question_context(current_q)
            tts_result = self.text_to_speech_hf(ai_message)
            
            return {
                "success": True,
                "status": "active",
                "question_id": current_q["id"],
                "question_number": self.current_question_index + 1,
                "total_questions": len(self.questions),
                "category": current_q["category"],
                "question": current_q["question"],
                "ai_message": ai_message,
                "ai_audio": tts_result,
                "difficulty": current_q["difficulty"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get question: {str(e)}"
            }
    
    def _generate_question_context(self, question_data: Dict) -> str:
        """Generate contextual message for each question"""
        context_intros = {
            "introduction": "Let's start with getting to know you better.",
            "technical_skills": "Now I'd like to explore your technical expertise.",
            "experience_based": "Tell me about your professional journey.",
            "problem_solving": "Let's discuss how you approach challenges.",
            "closing": "We're approaching the end of our interview."
        }
        
        intro = context_intros.get(question_data["category"], "Here's my next question for you.")
        return f"{intro} {question_data['question']}"
    
    def process_answer(self, answer_text: str, audio_duration: float = 0) -> Dict:
        """Process candidate answer with full error handling"""
        try:
            if not self.current_interview or self.current_question_index >= len(self.questions):
                return {"success": False, "error": "No active question"}
            
            if not answer_text.strip():
                return {"success": False, "error": "Please provide an answer"}
            
            current_question = self.questions[self.current_question_index]
            
            # Store conversation entry
            conversation_entry = {
                "question_id": current_question["id"],
                "question": current_question["question"],
                "category": current_question["category"],
                "candidate_answer": answer_text.strip(),
                "answer_duration": audio_duration,
                "timestamp": datetime.now(),
                "difficulty": current_question["difficulty"]
            }
            
            self.conversation_history.append(conversation_entry)
            self.current_question_index += 1
            
            # Save state
            self._save_interview_state()
            
            # Generate AI response
            ai_response = self._generate_ai_response(answer_text, current_question)
            tts_result = self.text_to_speech_hf(ai_response)
            
            # Check if interview complete
            if self.current_question_index >= len(self.questions):
                self.current_interview["status"] = "completed"
                self.current_interview["end_time"] = datetime.now()
                self._save_interview_state()
                
                completion_msg = "Thank you for your comprehensive responses! The interview is now complete. Let me analyze your answers and prepare your detailed report."
                completion_tts = self.text_to_speech_hf(completion_msg)
                
                return {
                    "success": True,
                    "status": "completed",
                    "ai_message": completion_msg,
                    "ai_audio": completion_tts,
                    "progress": 100,
                    "show_report": True
                }
            
            # Get next question
            next_question = self.get_next_question()
            
            return {
                "success": True,
                "status": "continuing",
                "ai_response": ai_response,
                "ai_audio": tts_result,
                "next_question": next_question,
                "progress": (self.current_question_index / len(self.questions)) * 100
            }
            
        except Exception as e:
            return {"success": False, "error": f"Processing failed: {str(e)}"}
    
    def _generate_ai_response(self, answer: str, question_data: Dict) -> str:
        """Generate natural AI responses"""
        responses = {
            "introduction": ["Thank you for that great introduction.", "I appreciate you sharing your background."],
            "technical_skills": ["That's valuable technical insight.", "Your technical experience sounds interesting."],
            "experience_based": ["That's a meaningful professional example.", "I can see how that experience shaped you."],
            "problem_solving": ["That's a thoughtful problem-solving approach.", "I like your analytical thinking."],
            "closing": ["Thank you for those final thoughts.", "Those are excellent closing comments."]
        }
        
        import random
        category_responses = responses.get(question_data["category"], ["Thank you for that thoughtful answer."])
        return random.choice(category_responses)
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate detailed interview report using Gemini"""
        if not self.conversation_history:
            return {"success": False, "error": "No interview data available"}
        
        try:
            # Format conversation for analysis
            conversation_text = self._format_conversation_for_analysis()
            
            prompt = f"""
            As an expert HR interviewer, analyze this interview conversation and provide a comprehensive assessment:
            
            INTERVIEW CONVERSATION:
            {conversation_text}
            
            CANDIDATE BACKGROUND:
            {json.dumps(self.current_interview.get('resume_analysis', {}), indent=2)}
            
            Provide a detailed JSON assessment with the following structure:
            {{
                "overall_performance": {{
                    "score": <0-100>,
                    "summary": "<comprehensive overall assessment>",
                    "key_strengths": ["strength1", "strength2", "strength3"],
                    "areas_for_improvement": ["area1", "area2"]
                }},
                "technical_assessment": {{
                    "score": <0-100>,
                    "technical_depth": "<assessment of technical knowledge>",
                    "problem_solving_ability": <0-100>,
                    "relevant_experience": <0-100>
                }},
                "communication_skills": {{
                    "score": <0-100>,
                    "clarity": <0-100>,
                    "articulation": <0-100>,
                    "confidence_level": <0-100>,
                    "listening_skills": <0-100>
                }},
                "behavioral_indicators": {{
                    "leadership_potential": <0-100>,
                    "teamwork": <0-100>,
                    "adaptability": <0-100>,
                    "motivation": <0-100>,
                    "cultural_fit": <0-100>
                }},
                "answer_quality": {{
                    "completeness": <0-100>,
                    "relevance": <0-100>,
                    "use_of_examples": <0-100>,
                    "depth_of_responses": <0-100>
                }},
                "hiring_recommendation": {{
                    "recommendation": "<strongly_recommend/recommend/consider/not_recommend>",
                    "confidence": <0-100>,
                    "reasoning": "<detailed explanation>",
                    "next_steps": "<suggested next steps>"
                }},
                "detailed_feedback": {{
                    "strengths_demonstrated": ["detailed strength 1", "detailed strength 2"],
                    "concerns_raised": ["concern 1", "concern 2"],
                    "development_areas": ["area 1", "area 2"],
                    "interview_highlights": ["highlight 1", "highlight 2"]
                }}
            }}
            
            Ensure all scores are realistic and well-justified based on the actual responses provided.
            """
            
            response = self.gemini_model.generate_content(prompt)
            report_data = self._parse_json_response(response.text)
            
            # Add metadata
            report_data["interview_metadata"] = {
                "candidate_name": self.current_interview["candidate_name"],
                "interview_date": self.current_interview["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "duration": str(self.current_interview.get("end_time", datetime.now()) - self.current_interview["start_time"]),
                "questions_answered": len(self.conversation_history),
                "interview_id": self.current_interview["id"]
            }
            
            # Generate summary audio
            summary = f"Your interview report is ready! Overall performance score: {report_data.get('overall_performance', {}).get('score', 'N/A')} out of 100. {report_data.get('overall_performance', {}).get('summary', '')[:200]}"
            tts_result = self.text_to_speech_hf(summary)
            report_data["summary_audio"] = tts_result
            
            return {"success": True, "report": report_data}
            
        except Exception as e:
            return {"success": False, "error": f"Report generation failed: {str(e)}"}
    
    def _format_conversation_for_analysis(self) -> str:
        """Format conversation for AI analysis"""
        formatted = []
        for i, entry in enumerate(self.conversation_history, 1):
            formatted.append(f"""
QUESTION {i} [{entry['category'].upper()}] - Difficulty: {entry['difficulty'].upper()}:
{entry['question']}

CANDIDATE RESPONSE:
{entry['candidate_answer']}

Response Duration: {entry.get('answer_duration', 'N/A')} seconds
---""")
        return "\n".join(formatted)
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from Gemini response"""
        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try direct JSON parsing
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError("No valid JSON found")
            
        except Exception as e:
            return {
                "error": f"Failed to parse report: {str(e)}",
                "raw_response": response_text[:500],
                "overall_performance": {"score": 50, "summary": "Report generation incomplete"}
            }
    
    def _save_interview_state(self):
        """Save interview to database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO interviews 
                (id, candidate_name, start_time, end_time, status, resume_analysis, questions, conversation_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_interview["id"],
                self.current_interview["candidate_name"],
                self.current_interview["start_time"].isoformat(),
                self.current_interview.get("end_time", {}).isoformat() if self.current_interview.get("end_time") else None,
                self.current_interview["status"],
                json.dumps(self.current_interview.get("resume_analysis", {})),
                json.dumps(self.current_interview.get("questions", [])),
                json.dumps([{**entry, "timestamp": entry["timestamp"].isoformat()} for entry in self.conversation_history])
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving interview: {e}")