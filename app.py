from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import time
from werkzeug.utils import secure_filename
from models.resume_matcher import ResumeMatcher
from models.question_generator import QuestionGenerator
from models.interview import InterviewSimulator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize with error handling
try:
    resume_matcher = ResumeMatcher()
    question_generator = QuestionGenerator()
    interview_simulator = InterviewSimulator()
    print("All components initialized successfully")
except Exception as e:
    print(f"Error initializing components: {e}")
    # Initialize with fallback
    resume_matcher = ResumeMatcher()
    question_generator = QuestionGenerator()
    interview_simulator = None

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return redirect(request.url)
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze resume using Gemini API
        try:
            analysis_results = resume_matcher.analyze_resume(file_path, job_description)
            
            # Store analysis results in session or pass to next step
            return render_template('results.html', 
                                 filename=filename,
                                 job_description=job_description,
                                 analysis=analysis_results,
                                 file_path=file_path)
        except Exception as e:
            return render_template('error.html', 
                                 error=f"Analysis failed: {str(e)}")
    
    return redirect(request.url)

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Generate interview questions based on resume analysis"""
    try:
        filename = request.form.get('filename')
        job_description = request.form.get('job_description')
        file_path = request.form.get('file_path')
        
        # Validate file path exists
        if not file_path or not os.path.exists(file_path):
            # Reconstruct file path if missing
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Get analysis results again or from session
        analysis_results = resume_matcher.analyze_resume(file_path, job_description)
        
        # Extract resume text for question generation
        resume_text = resume_matcher.extract_text_from_pdf(file_path)
        
        # Generate interview questions
        questions_data = question_generator.generate_interview_questions(
            analysis_results, resume_text, job_description
        )
        
        # Get sequential question list and summary
        question_sequence = question_generator.get_question_sequence(questions_data)
        interview_summary = question_generator.get_interview_summary(questions_data)
        
        return render_template('interview_questions.html',
                             filename=filename,
                             analysis=analysis_results,
                             questions=question_sequence,
                             summary=interview_summary,
                             questions_data=questions_data)
                             
    except Exception as e:
        return render_template('error.html',
                             error=f"Question generation failed: {str(e)}")

@app.route('/start-interview', methods=['POST'])
def start_interview():
    """Start a new interview session"""
    if not interview_simulator:
        return {"success": False, "error": "Interview simulator not available"}
    
    try:
        data = request.get_json()
        questions_data = data.get('questions_data', {})
        candidate_name = data.get('candidate_name', 'Candidate')
        resume_analysis = data.get('resume_analysis', {})
        
        result = interview_simulator.start_interview(questions_data, candidate_name, resume_analysis)
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/get-question', methods=['GET'])
def get_question():
    """Get the current question"""
    if not interview_simulator:
        return jsonify({"success": False, "error": "Interview simulator not available"})
    
    try:
        question_data = interview_simulator.get_next_question()
        return jsonify({"success": True, "data": question_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    """Submit candidate answer and get next question"""
    if not interview_simulator:
        return jsonify({"success": False, "error": "Interview simulator not available"})
    
    try:
        data = request.get_json()
        answer_text = data.get('answer_text')
        audio_duration = data.get('audio_duration', 0)
        
        result = interview_simulator.process_candidate_answer(answer_text, audio_duration)
        return jsonify({"success": True, "data": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe uploaded audio"""
    if not interview_simulator:
        return jsonify({"success": False, "error": "Transcription not available"})
    
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file"})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        # Save temporary file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_audio_{int(time.time())}.wav")
        audio_file.save(temp_path)
        
        # Transcribe
        transcription = interview_simulator.transcribe_audio(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({"success": True, "transcription": transcription})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/end-interview', methods=['POST'])
def end_interview():
    """End the current interview"""
    if not interview_simulator:
        return jsonify({"success": False, "error": "Interview simulator not available"})
    
    try:
        result = interview_simulator.end_interview()
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate final interview report"""
    if not interview_simulator:
        return jsonify({"success": False, "error": "Interview simulator not available"})
    
    try:
        report = interview_simulator.generate_final_report()
        return jsonify({"success": True, "data": report})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/interview-simulator')
def interview_simulator_page():
    """Render the interview simulator page"""
    return render_template('interview_simulator.html')

@app.route('/interview-report')
def interview_report():
    """Render the interview report page"""
    return render_template('interview_report.html')

if __name__ == '__main__':
    app.run(debug=True)