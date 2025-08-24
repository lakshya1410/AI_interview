# AI Interview System

An intelligent interview system that uses AI to generate questions and evaluate candidates based on their resumes.

## Features

- Resume upload and parsing
- AI-powered question generation
- Interactive interview simulation
- Resume-question matching
- Interview results analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lakshya1410/AI_interview.git
cd AI_interview
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `GROQ_API_KEY`: Your Groq API key

5. Run the application:
```bash
python app.py
```

## Project Structure

- `app.py` - Main Flask application
- `models/` - AI models and business logic
  - `interview.py` - Interview management
  - `question_generator.py` - AI question generation
  - `resume_matcher.py` - Resume parsing and matching
- `templates/` - HTML templates
- `data/` - Database files
- `uploads/` - Resume upload directory (not tracked)

## Requirements

- Python 3.8+
- Flask
- Google Generative AI
- Groq API
- PyPDF2
- SQLite

## Security

Make sure to:
- Never commit your `.env` file
- Keep your API keys secure
- Use environment variables for sensitive data

## License

This project is for educational purposes.
