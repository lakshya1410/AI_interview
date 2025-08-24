import os
import json
import time
import base64
import sqlite3
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class InterviewSimulator:
    def __init__(self):
        # Configure APIs
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize Groq client for STT only
        self.groq_client = None
        self.tts_available = False
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                print("Groq client initialized successfully for STT")
        except Exception as e:
            print(f"Warning: Could not initialize Groq client: {e}")
            print("Speech-to-text functionality will be limited")
        
        # Initialize database
        self.db_path = Path("c:/Users/LAKSHYA/Desktop/AI_interview/data/interviews.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Audio file management
        self.temp_audio_files = set()
        
        # Interview state
        self.current_interview = None
        self.conversation_history = []
        self.current_question_index = 0
        self.start_time = None
        self.questions = []
        self.audio_enabled = True
        self.video_visible = True
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interview_id TEXT,
                    question_id INTEGER,
                    question TEXT,
                    category TEXT,
                    candidate_answer TEXT,
                    answer_duration REAL,
                    timestamp TEXT,
                    difficulty TEXT,
                    audio_file_path TEXT,
                    FOREIGN KEY (interview_id) REFERENCES interviews (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Database initialized successfully")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def save_interview_state(self):
        """Save current interview state to database"""
        if not self.current_interview:
            return False
            
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO interviews 
                (id, candidate_name, start_time, end_time, status, resume_analysis, questions, conversation_history, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_interview["id"],
                self.current_interview["candidate_name"],
                self.current_interview["start_time"].isoformat(),
                self.current_interview.get("end_time", {}).isoformat() if self.current_interview.get("end_time") else None,
                self.current_interview["status"],
                json.dumps(self.current_interview.get("resume_analysis", {})),
                json.dumps(self.current_interview.get("questions", [])),
                json.dumps([{**entry, "timestamp": entry["timestamp"].isoformat() if isinstance(entry["timestamp"], datetime) else entry["timestamp"]} for entry in self.conversation_history]),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving interview state: {e}")
            return False
    
    def load_interview_state(self, interview_id: str) -> bool:
        """Load interview state from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,))
            row = cursor.fetchone()
            
            if row:
                # Reconstruct interview state
                self.current_interview = {
                    "id": row[0],
                    "candidate_name": row[1],
                    "start_time": datetime.fromisoformat(row[2]),
                    "end_time": datetime.fromisoformat(row[3]) if row[3] else None,
                    "status": row[4],
                    "resume_analysis": json.loads(row[5]) if row[5] else {},
                    "questions": json.loads(row[6]) if row[6] else []
                }
                
                self.conversation_history = json.loads(row[7]) if row[7] else []
                # Convert timestamp strings back to datetime objects
                for entry in self.conversation_history:
                    entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                
                self.questions = self.current_interview["questions"]
                self.current_question_index = len(self.conversation_history)
                
                conn.close()
                return True
            
            conn.close()
            return False
        except Exception as e:
            print(f"Error loading interview state: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        for file_path in self.temp_audio_files.copy():
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                self.temp_audio_files.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up temp file {file_path}: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_temp_files()

    def start_interview(self, questions_data: Dict, candidate_name: str, resume_analysis: Dict) -> Dict:
        """Initialize a new interview session with enhanced error handling"""
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
            
            # Save to database immediately
            self.save_interview_state()
            
            # Generate opening message
            opening_message = self._generate_opening_message(candidate_name)
            
            # Generate TTS for opening message (with fallback)
            audio_data = self.text_to_speech(opening_message)
            
            return {
                "interview_id": self.current_interview["id"],
                "status": "started",
                "opening_message": opening_message,
                "total_questions": len(self.questions),
                "ai_message": opening_message,
                "ai_audio": audio_data,
                "video_visible": True,
                "success": True
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to start interview: {str(e)}",
                "success": False
            }
    
    def _extract_questions(self, questions_data: Dict) -> List[Dict]:
        """Extract questions from questions_data in sequence"""
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
                            "time_allocation": q.get('time_allocation', '2-3 minutes'),
                            "purpose": q.get('purpose', ''),
                            "expected_topics": q.get('expected_topics', []),
                            "follow_up": q.get('follow_up', '')
                        })
                        question_id += 1
        
        return questions
    
    def _generate_opening_message(self, candidate_name: str) -> str:
        """Generate personalized opening message"""
        return f"Hello {candidate_name}! Welcome to your AI interview. I'm your virtual interviewer today. We'll be going through {len(self.questions)} questions covering various aspects of your background and skills. Please feel comfortable and answer naturally. Are you ready to begin?"
    
    def get_next_question(self) -> Dict:
        """Get the next question in the interview"""
        if self.current_question_index >= len(self.questions):
            completion_message = "Thank you for completing the interview! Let me analyze your responses and generate your report."
            audio_data = self.text_to_speech(completion_message)
            return {
                "status": "completed", 
                "message": completion_message,
                "ai_audio": audio_data,
                "video_visible": True,
                "show_report": True
            }
        
        current_q = self.questions[self.current_question_index]
        
        # Add context for AI response
        ai_message = self._generate_question_context(current_q)
        
        # Generate TTS for the question
        audio_data = self.text_to_speech(ai_message)
        
        return {
            "status": "active",
            "question_id": current_q["id"],
            "question_number": self.current_question_index + 1,
            "total_questions": len(self.questions),
            "category": current_q["category"],
            "question": current_q["question"],
            "time_allocation": current_q["time_allocation"],
            "ai_message": ai_message,
            "ai_audio": audio_data,
            "difficulty": current_q["difficulty"],
            "video_visible": True
        }
    
    def _generate_question_context(self, question_data: Dict) -> str:
        """Generate contextual AI message for asking question"""
        context_messages = {
            "introduction": "Let's start with an introductory question.",
            "technical_skills": "Now I'd like to explore your technical background.",
            "experience_based": "Tell me about your professional experience.",
            "problem_solving": "Let's discuss a problem-solving scenario.",
            "closing": "We're nearing the end of our interview."
        }
        
        context = context_messages.get(question_data["category"], "Here's my next question.")
        return f"{context} {question_data['question']}"
    
    def process_candidate_answer(self, answer_text: str, audio_duration: float = None) -> Dict:
        """Process candidate's answer with enhanced error handling and persistence"""
        try:
            if not self.current_interview:
                raise ValueError("No active interview session")
                
            if self.current_question_index >= len(self.questions):
                raise ValueError("Interview already completed")
                
            if not answer_text.strip():
                raise ValueError("Empty answer provided")
            
            current_question = self.questions[self.current_question_index]
            
            # Store the conversation entry
            conversation_entry = {
                "question_id": current_question["id"],
                "question": current_question["question"],
                "category": current_question["category"],
                "candidate_answer": answer_text.strip(),
                "answer_duration": audio_duration,
                "timestamp": datetime.now(),
                "difficulty": current_question["difficulty"],
                "answer_type": "voice" if audio_duration else "text"
            }
            
            self.conversation_history.append(conversation_entry)
            
            # Move to next question
            self.current_question_index += 1
            
            # Save state immediately after each answer
            self.save_interview_state()
            
            # Generate AI response
            ai_response = self._generate_enhanced_ai_response(answer_text, current_question)
            ai_audio = self.text_to_speech(ai_response)
            
            # Check if interview is complete
            if self.current_question_index >= len(self.questions):
                completion_message = "Thank you for your responses. The interview is now complete. Let me analyze your answers and prepare your report."
                completion_audio = self.text_to_speech(completion_message)
                
                # Mark interview as completed and save
                self.current_interview["status"] = "completed"
                self.current_interview["end_time"] = datetime.now()
                self.save_interview_state()
                
                return {
                    "status": "completed",
                    "ai_message": completion_message,
                    "ai_audio": completion_audio,
                    "progress": 100,
                    "video_visible": True,
                    "show_report": True,
                    "conversation_entry": conversation_entry
                }
            
            # Get next question
            next_question = self.get_next_question()
            
            return {
                "status": "continuing",
                "ai_response": ai_response,
                "ai_audio": ai_audio,
                "next_question": next_question,
                "progress": (self.current_question_index / len(self.questions)) * 100,
                "video_visible": True,
                "conversation_entry": conversation_entry
            }
            
        except ValueError as e:
            return {"status": "error", "error": str(e), "video_visible": True}
        except Exception as e:
            return {"status": "error", "error": f"Failed to process answer: {str(e)}", "video_visible": True}

    def _generate_enhanced_ai_response(self, answer: str, question_data: Dict) -> str:
        """Generate more natural AI interviewer responses based on context"""
        category = question_data.get("category", "")
        
        responses_by_category = {
            "introduction": [
                "Thank you for that introduction.",
                "Great to learn more about your background.",
                "That gives me a good understanding of who you are.",
                "I appreciate you sharing that background information."
            ],
            "technical_skills": [
                "That's valuable technical insight.",
                "I appreciate the technical details you've shared.",
                "Your technical experience is quite interesting.",
                "Thank you for explaining your technical approach."
            ],
            "experience_based": [
                "That sounds like meaningful professional experience.",
                "I can see how that experience has shaped your perspective.",
                "Thank you for sharing that professional example.",
                "That's a great example from your experience."
            ],
            "problem_solving": [
                "That's an interesting problem-solving approach.",
                "I like how you think through challenges systematically.",
                "Your analytical thinking process is clear.",
                "That demonstrates good problem-solving skills."
            ],
            "closing": [
                "Thank you for those thoughtful final comments.",
                "That wraps up our discussion very well.",
                "I appreciate your time and comprehensive responses.",
                "Those are excellent closing thoughts."
            ]
        }
        
        import random
        category_responses = responses_by_category.get(category, [
            "Thank you for that thoughtful answer.",
            "I appreciate your detailed response.",
            "That's very helpful information.",
            "Thank you for sharing that with me."
        ])
        
        return random.choice(category_responses)
    
    def text_to_speech(self, text: str) -> Dict:
        """Text-to-speech with proper fallback handling"""
        if not self.audio_enabled:
            return {
                "audio_data": None,
                "format": None,
                "success": False,
                "text": text,
                "message": "Audio disabled by user"
            }
        
        # Since Groq doesn't have TTS, we'll use browser-based TTS or disable it
        try:
            # Fallback: Return text for browser-based TTS
            return {
                "audio_data": None,
                "format": "text",
                "success": True,
                "text": text,
                "use_browser_tts": True,
                "message": "Using browser text-to-speech"
            }
        except Exception as e:
            return {
                "audio_data": None,
                "format": None,
                "success": False,
                "error": str(e),
                "text": text,
                "fallback": True
            }

    def transcribe_audio(self, audio_file_path: str) -> Dict:
        """Enhanced audio transcription with comprehensive error handling"""
        if not self.groq_client:
            return {
                "text": "",
                "success": False,
                "error": "Audio transcription service not available. Please type your response.",
                "fallback_to_text": True
            }
        
        try:
            # Validate file exists and has content
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            if file_size > 25 * 1024 * 1024:  # 25MB limit
                raise ValueError("Audio file too large (max 25MB)")
            
            # Validate audio format
            valid_formats = ['.wav', '.mp3', '.mp4', '.m4a', '.webm']
            file_ext = Path(audio_file_path).suffix.lower()
            if file_ext not in valid_formats:
                raise ValueError(f"Unsupported audio format: {file_ext}")
            
            with open(audio_file_path, "rb") as file:
                # Use Groq's Whisper API correctly
                transcription = self.groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    language="en"
                )
                
                # Handle response properly
                if hasattr(transcription, 'text'):
                    text = transcription.text.strip()
                    confidence = getattr(transcription, 'confidence', None)
                else:
                    text = str(transcription).strip()
                    confidence = None
                
                if not text:
                    raise ValueError("No speech detected in audio")
                
                return {
                    "text": text,
                    "success": True,
                    "duration": self._get_audio_duration(audio_file_path),
                    "confidence": confidence,
                    "file_size": file_size
                }
                
        except FileNotFoundError as e:
            return {"text": "", "success": False, "error": str(e), "fallback_to_text": True}
        except ValueError as e:
            return {"text": "", "success": False, "error": str(e), "fallback_to_text": True}
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            print(error_msg)
            return {"text": "", "success": False, "error": error_msg, "fallback_to_text": True}
    
    def _get_audio_duration(self, audio_file_path: str) -> float:
        """Get audio file duration with better format support"""
        try:
            file_ext = Path(audio_file_path).suffix.lower()
            
            if file_ext == '.wav':
                import wave
                with wave.open(audio_file_path, 'r') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    return frames / float(rate)
            else:
                # For other formats, try to use a more general approach
                # This is a fallback - in production you might want to use librosa or similar
                file_size = os.path.getsize(audio_file_path)
                # Rough estimate: assume ~128kbps bitrate
                estimated_duration = file_size / (128 * 1024 / 8)  # seconds
                return min(estimated_duration, 300)  # Cap at 5 minutes
                
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0

    def save_audio_recording(self, audio_data: bytes, format: str = "wav") -> Optional[str]:
        """Save recorded audio to file with proper cleanup tracking"""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{format}",
                prefix="interview_audio_"
            )
            
            temp_file.write(audio_data)
            temp_file.close()
            
            # Track for cleanup
            self.temp_audio_files.add(temp_file.name)
            
            return temp_file.name
        except Exception as e:
            print(f"Error saving audio recording: {e}")
            return None

    def process_candidate_voice_answer(self, audio_file_path: str) -> Dict:
        """Process candidate's voice answer with comprehensive error handling"""
        try:
            # Validate input
            if not audio_file_path or not os.path.exists(audio_file_path):
                return {
                    "status": "error",
                    "error": "Invalid audio file",
                    "video_visible": True,
                    "fallback_to_text": True
                }
            
            # Transcribe the audio
            transcription_result = self.transcribe_audio(audio_file_path)
            
            if not transcription_result["success"]:
                return {
                    "status": "error",
                    "error": transcription_result["error"],
                    "video_visible": True,
                    "fallback_to_text": transcription_result.get("fallback_to_text", False)
                }
            
            answer_text = transcription_result["text"]
            audio_duration = transcription_result.get("duration", 0.0)
            
            if not answer_text.strip():
                return {
                    "status": "error",
                    "error": "No speech detected. Please speak clearly or use text input.",
                    "video_visible": True,
                    "fallback_to_text": True
                }
            
            # Store audio file data for chat display
            audio_base64 = None
            try:
                with open(audio_file_path, "rb") as f:
                    audio_content = f.read()
                    audio_base64 = base64.b64encode(audio_content).decode()
            except Exception as e:
                print(f"Error reading audio file for chat: {e}")
            
            # Process the answer
            result = self.process_candidate_answer(answer_text, audio_duration)
            
            # Add audio data to result for chat display
            if result.get("status") != "error":
                result["candidate_audio"] = {
                    "audio_data": audio_base64,
                    "format": Path(audio_file_path).suffix.lower().replace('.', ''),
                    "transcribed_text": answer_text,
                    "duration": audio_duration,
                    "confidence": transcription_result.get("confidence")
                }
                
                # Update conversation history with audio info
                if self.conversation_history:
                    self.conversation_history[-1]["answer_type"] = "voice"
                    self.conversation_history[-1]["audio_file_path"] = audio_file_path
                    
                # Save updated state
                self.save_interview_state()
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to process voice answer: {str(e)}",
                "video_visible": True,
                "fallback_to_text": True
            }

    def end_interview(self) -> Dict:
        """End the interview and prepare for report generation"""
        if not self.current_interview:
            return {"error": "No active interview"}
        
        self.current_interview["end_time"] = datetime.now()
        self.current_interview["status"] = "completed"
        
        ending_message = "Interview ended. Generating your comprehensive report..."
        ending_audio = self.text_to_speech(ending_message)
        
        return {
            "status": "ended",
            "interview_id": self.current_interview["id"],
            "total_duration": str(self.current_interview["end_time"] - self.current_interview["start_time"]),
            "questions_answered": len(self.conversation_history),
            "message": ending_message,
            "ai_audio": ending_audio,
            "video_visible": True,
            "show_report": True
        }
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive interview report using Gemini"""
        if not self.conversation_history:
            return {"error": "No interview data available"}
        
        # Prepare conversation data for analysis
        conversation_text = self._format_conversation_for_analysis()
        
        prompt = f"""
        Analyze this interview conversation and generate a comprehensive report:
        
        INTERVIEW DATA:
        {conversation_text}
        
        RESUME ANALYSIS CONTEXT:
        {json.dumps(self.current_interview.get('resume_analysis', {}), indent=2)}
        
        Generate a detailed interview assessment report in the following JSON format:
        {{
            "overall_performance": {{
                "score": <0-100>,
                "summary": "<overall assessment>"
            }},
            "technical_knowledge": {{
                "score": <0-100>,
                "strengths": ["<strength1>", "<strength2>"],
                "weaknesses": ["<weakness1>", "<weakness2>"],
                "assessment": "<detailed technical assessment>"
            }},
            "communication_skills": {{
                "score": <0-100>,
                "clarity": <0-100>,
                "articulation": <0-100>,
                "confidence": <0-100>,
                "assessment": "<communication assessment>"
            }},
            "problem_solving": {{
                "score": <0-100>,
                "analytical_thinking": <0-100>,
                "creativity": <0-100>,
                "approach": "<problem-solving approach assessment>"
            }},
            "experience_relevance": {{
                "score": <0-100>,
                "alignment": "<how well experience aligns with role>",
                "examples": ["<relevant example1>", "<relevant example2>"]
            }},
            "behavioral_assessment": {{
                "leadership": <0-100>,
                "teamwork": <0-100>,
                "adaptability": <0-100>,
                "motivation": <0-100>
            }},
            "answer_quality": {{
                "completeness": <0-100>,
                "relevance": <0-100>,
                "depth": <0-100>,
                "examples_provided": <number>
            }},
            "areas_of_concern": [
                "<concern1>",
                "<concern2>"
            ],
            "recommendations": [
                "<recommendation1>",
                "<recommendation2>"
            ],
            "interview_flow": {{
                "engagement_level": <0-100>,
                "response_time": "<assessment of response timing>",
                "question_understanding": <0-100>
            }},
            "final_verdict": {{
                "recommendation": "<hire/not_hire/maybe>",
                "reasoning": "<detailed reasoning>",
                "fit_score": <0-100>
            }}
        }}
        
        Consider:
        1. Quality and relevance of answers
        2. Technical depth and accuracy
        3. Communication clarity and confidence
        4. Problem-solving approach
        5. Cultural and role fit
        6. Overall interview performance
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            report_data = self._parse_report_response(response.text)
            
            # Add metadata
            report_data["interview_metadata"] = {
                "candidate_name": self.current_interview.get("candidate_name"),
                "interview_date": self.current_interview["start_time"].isoformat(),
                "duration": str(self.current_interview["end_time"] - self.current_interview["start_time"]),
                "questions_answered": len(self.conversation_history),
                "interview_id": self.current_interview["id"]
            }
            
            # Generate audio summary of the report
            summary_text = f"Your interview report has been generated. Overall performance score: {report_data.get('overall_performance', {}).get('score', 'N/A')} out of 100. {report_data.get('overall_performance', {}).get('summary', '')}"
            report_audio = self.text_to_speech(summary_text)
            
            report_data["report_audio"] = report_audio
            report_data["video_visible"] = True
            report_data["report_ready"] = True
            
            return report_data
            
        except Exception as e:
            error_message = f"Report generation failed: {str(e)}"
            error_audio = self.text_to_speech(error_message)
            return {
                "error": error_message,
                "error_audio": error_audio,
                "video_visible": True
            }
    
    def _format_conversation_for_analysis(self) -> str:
        """Format conversation history for AI analysis"""
        formatted = []
        for i, entry in enumerate(self.conversation_history, 1):
            formatted.append(f"""
Q{i} [{entry['category'].upper()}] - {entry['difficulty'].upper()}:
{entry['question']}

CANDIDATE ANSWER:
{entry['candidate_answer']}

Duration: {entry.get('answer_duration', 'N/A')} seconds
---""")
        
        return "\n".join(formatted)
    
    def _parse_report_response(self, response_text: str) -> Dict:
        """Parse Gemini response and extract report JSON"""
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
            return {
                "error": "Failed to parse report",
                "raw_response": response_text[:1000],
                "overall_performance": {"score": 50, "summary": "Analysis incomplete"}
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def get_interview_status(self) -> Dict:
        """Get current interview status"""
        if not self.current_interview:
            return {"status": "no_active_interview", "video_visible": False}
        
        return {
            "status": self.current_interview["status"],
            "progress": (self.current_question_index / len(self.questions)) * 100,
            "current_question": self.current_question_index + 1,
            "total_questions": len(self.questions),
            "questions_answered": len(self.conversation_history),
            "video_visible": True,
            "audio_enabled": self.audio_enabled
        }
    
    def toggle_audio(self) -> Dict:
        """Toggle audio on/off"""
        self.audio_enabled = not self.audio_enabled
        return {
            "audio_enabled": self.audio_enabled,
            "message": f"Audio {'enabled' if self.audio_enabled else 'disabled'}"
        }
    
    def ensure_video_visibility(self) -> Dict:
        """Ensure video feed remains visible"""
        self.video_visible = True
        return {
            "video_visible": True,
            "message": "Video feed is active"
        }
    
    def get_chat_history_for_display(self) -> List[Dict]:
        """Get formatted chat history for UI display"""
        chat_history = []
        
        for i, entry in enumerate(self.conversation_history):
            # Add question from AI
            chat_history.append({
                "type": "ai_question",
                "content": entry["question"],
                "timestamp": entry["timestamp"],
                "category": entry["category"]
            })
            
            # Add candidate answer
            chat_history.append({
                "type": "candidate_answer", 
                "content": entry["candidate_answer"],
                "timestamp": entry["timestamp"],
                "duration": entry.get("answer_duration"),
                "answer_type": entry.get("answer_type", "text")
            })
        
        return chat_history

    def export_interview_report(self, interview_id: str, format: str = "json") -> Optional[str]:
        """Export interview report to file"""
        try:
            report_data = self.generate_final_report()
            if "error" in report_data:
                return None
            
            export_dir = Path("c:/Users/LAKSHYA/Desktop/AI_interview/exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate_name = self.current_interview.get("candidate_name", "unknown").replace(" ", "_")
            
            if format.lower() == "json":
                filename = f"interview_report_{candidate_name}_{timestamp}.json"
                filepath = export_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                
                return str(filepath)
            
            # Add more export formats as needed (PDF, HTML, etc.)
            return None
            
        except Exception as e:
            print(f"Error exporting report: {e}")
            return None

    def get_saved_interviews(self) -> List[Dict]:
        """Get list of saved interviews from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, candidate_name, start_time, status, 
                       (SELECT COUNT(*) FROM conversation_entries WHERE interview_id = interviews.id) as question_count
                FROM interviews 
                ORDER BY start_time DESC
            ''')
            
            interviews = []
            for row in cursor.fetchall():
                interviews.append({
                    "id": row[0],
                    "candidate_name": row[1],
                    "start_time": row[2],
                    "status": row[3],
                    "questions_answered": row[4]
                })
            
            conn.close()
            return interviews
            
        except Exception as e:
            print(f"Error getting saved interviews: {e}")
            return []
