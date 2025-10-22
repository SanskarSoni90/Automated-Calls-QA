#!/usr/bin/env python3
"""
Automated QA Engine for Call Analysis
Runs hourly via GitHub Actions to analyze recent call recordings
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import requests
from requests.auth import HTTPBasicAuth
import time

# AI Service Imports
try:
    import assemblyai as aai
    import openai
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qa_engine.log')
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EXOTEL_ACCOUNT_SID = os.environ.get("EXOTEL_ACCOUNT_SID")
EXOTEL_API_KEY = os.environ.get("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.environ.get("EXOTEL_API_TOKEN")
GSPREAD_CREDENTIALS = os.environ.get("GSPREAD_CREDENTIALS")
EXOTEL_SUBDOMAIN = "api.exotel.com"

# Google Sheet Configuration
SHEET_ID = "1oYtYT7HkvpHL9fjHxf9upzFKzuNlrdh39d65sJM-c1U"  # Your specific sheet
SHEET_NAME = None  # Will open by ID instead of name

# Timezone Configuration
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)  # IST
TIMEZONE_NAME = "IST"

def get_local_time() -> datetime:
    """Get current time in configured timezone"""
    from datetime import timezone
    LOCAL_TZ = timezone(TIMEZONE_OFFSET)
    return datetime.now(LOCAL_TZ)

def format_ist_timestamp(dt: datetime) -> str:
    """Format datetime to string with timezone indicator"""
    return dt.strftime('%Y-%m-%d %H:%M:%S') + f' {TIMEZONE_NAME}'

# Initialize AI services
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

OPENAI_CLIENT = None
if OPENAI_API_KEY:
    OPENAI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)

ANALYSIS_MODEL = "gpt-4-turbo-preview"

# --- RUBRIC ---
RUBRIC = {
    "call_opening": 5,
    "call_closing": 5,
    "technically_legally_correct": 15,
    "all_questions_addressed": 10,
    "expectation_setting": 10,
    "process_adherence": 5,
    "vocabulary_grammar_pronunciation": 10,
    "fillers_fumbling_clarity": 10,
    "energy_tone_modulation": 10,
    "active_listening_interruptions": 10,
    "simplifying_answers": 10,
    "empathy": 10,
}
MAX_SCORE = sum(RUBRIC.values())


def validate_environment():
    """Validate all required environment variables are set"""
    required_vars = {
        "ASSEMBLYAI_API_KEY": ASSEMBLYAI_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "EXOTEL_ACCOUNT_SID": EXOTEL_ACCOUNT_SID,
        "EXOTEL_API_KEY": EXOTEL_API_KEY,
        "EXOTEL_API_TOKEN": EXOTEL_API_TOKEN,
        "GSPREAD_CREDENTIALS": GSPREAD_CREDENTIALS
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        return False
    
    logger.info("✅ All environment variables validated")
    return True


def fetch_google_doc_content(doc_id: str) -> str:
    """Fetch KB content from Google Doc"""
    try:
        creds_dict = json.loads(GSPREAD_CREDENTIALS)
        scopes = [
            "https://www.googleapis.com/auth/documents.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        
        from googleapiclient.discovery import build
        service = build('docs', 'v1', credentials=creds)
        document = service.documents().get(documentId=doc_id).execute()
        
        content = ""
        for element in document.get('body', {}).get('content', []):
            if 'paragraph' in element:
                for text_run in element['paragraph'].get('elements', []):
                    if 'textRun' in text_run:
                        content += text_run['textRun'].get('content', '')
        
        return content.strip()
    except Exception as e:
        logger.error(f"Error fetching KB: {e}")
        return "KB unavailable"


def get_kb_context() -> str:
    """Get knowledge base context"""
    doc_id = "1mrGpP_3TBRIGeQ7ekEq5o27MrO4XkLZt6_9faM_E8ZM"
    return fetch_google_doc_content(doc_id)


def ensure_sheet_headers(sheet):
    """Ensure the sheet has proper headers, create them if missing"""
    try:
        # Check if headers exist
        existing_headers = sheet.row_values(1)
        
        # Define complete header structure
        headers = [
            "Analysis Timestamp",
            "Call SID",
            "Call Date",
            "Start Time",
            "End Time",
            "Duration (seconds)",
            "From Number",
            "To Number",
            "Status",
            "Direction",
            "Answered By",
            "Price",
            "Recording URL",
            "Total Score",
            "Score Percentage",
            "Performance Level",
            "Conversation Tone",
            "Languages",
            "Unanswered Questions",
            "Emotional Arc",
        ]
        
        # Add headers for each rubric parameter (4 columns each)
        rubric_params = [
            "Call Opening",
            "Call Closing",
            "Technically/Legally Correct",
            "All Questions Addressed",
            "Expectation Setting",
            "Process Adherence",
            "Vocabulary/Grammar/Pronunciation",
            "Fillers/Fumbling/Clarity",
            "Energy/Tone/Modulation",
            "Active Listening/Interruptions",
            "Simplifying Answers",
            "Empathy"
        ]
        
        for param in rubric_params:
            headers.extend([
                f"{param} - Score",
                f"{param} - Justification",
                f"{param} - Transcript Quote",
                f"{param} - KB Recommended Answer"
            ])
        
        # Add final column
        headers.append("Full Transcript")
        
        # If no headers or incomplete headers, set them
        if not existing_headers or len(existing_headers) < len(headers):
            logger.info("📋 Setting up sheet headers...")
            sheet.update('A1', [headers], value_input_option='USER_ENTERED')
            
            # Format header row (bold, freeze)
            sheet.format('A1:ZZ1', {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
            sheet.freeze(rows=1)
            
            logger.info(f"✅ Headers created: {len(headers)} columns")
        else:
            logger.info(f"✅ Headers already exist: {len(existing_headers)} columns")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up headers: {e}")
        traceback.print_exc()
        return False


def get_already_processed_calls(sheet) -> set:
    """Get set of already processed Call SIDs from Google Sheet"""
    try:
        # Get all values from column B (Call SID column, row 1 is header)
        all_values = sheet.col_values(2)
        
        # Skip header (row 1) and get unique SIDs
        if len(all_values) > 1:
            processed_sids = set(all_values[1:])  # Skip header
            logger.info(f"Found {len(processed_sids)} already processed calls")
            return processed_sids
        else:
            logger.info("No processed calls found (sheet is empty or only has headers)")
            return set()
            
    except Exception as e:
        logger.warning(f"Could not fetch processed calls: {e}")
        return set()


def fetch_last_hour_recordings() -> List[Dict[str, Any]]:
    """Fetch all recordings from the last hour"""
    try:
        # Get current local time and calculate time range
        end_time = get_local_time()
        start_time = end_time - timedelta(hours=1)
        
        # Format for Exotel API
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Current {TIMEZONE_NAME} time: {format_ist_timestamp(end_time)}")
        logger.info(f"Fetching recordings from {format_ist_timestamp(start_time)} to {format_ist_timestamp(end_time)}")
        
        url = f"https://{EXOTEL_SUBDOMAIN}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json"
        
        date_filter = f"gte:{start_str};lte:{end_str}"
        params = {
            "PageSize": 100,
            "DateCreated": date_filter,
            "SortBy": "DateCreated:desc",
            "Status": "completed"
        }
        
        all_calls = []
        seen_sids = set()  # FIX: Track unique Call SIDs
        page = 0
        max_pages = 50
        
        while page < max_pages:
            params["Page"] = page
            
            response = requests.get(
                url,
                params=params,
                auth=HTTPBasicAuth(EXOTEL_API_KEY, EXOTEL_API_TOKEN),
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Exotel API error: {response.status_code}")
                break
            
            data = response.json()
            calls = data.get("Calls", [])
            
            if not calls:
                break
            
            # FIX: Filter calls with recordings AND deduplicate by SID
            for call in calls:
                call_sid = call.get("Sid")
                has_recording = call.get("RecordingUrl") or call.get("PreSignedRecordingUrl")
                
                if has_recording and call_sid and call_sid not in seen_sids:
                    all_calls.append(call)
                    seen_sids.add(call_sid)
            
            logger.info(f"Page {page}: Found {len(calls)} calls, {len(all_calls)} unique with recordings so far")
            
            # Check if there are more pages
            metadata = data.get("Metadata", {})
            if not metadata.get("NextPageUri"):
                break
            
            page += 1
        
        logger.info(f"✅ Total unique recordings fetched: {len(all_calls)}")
        return all_calls
        
    except Exception as e:
        logger.error(f"Error fetching recordings: {e}")
        traceback.print_exc()
        return []


def transcribe_audio(audio_url: str) -> Tuple[Optional[str], List[str], List[Dict]]:
    """Transcribe audio and extract sentiment data"""
    try:
        logger.info("Starting transcription...")
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2,
            punctuate=True,
            format_text=True,
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_url, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"Transcription failed: {transcript.error}")
            return None, [], []
        
        # Format transcript
        raw_text = ""
        if transcript.utterances:
            raw_text = "\n".join([f"Speaker {u.speaker}: {u.text}" for u in transcript.utterances])
        else:
            raw_text = transcript.text or ""
        
        # Detect languages
        detected_langs = []
        if hasattr(transcript, 'json_response') and transcript.json_response:
            lang_code = transcript.json_response.get("language_code")
            if lang_code:
                detected_langs = [lang_code]
        
        if not detected_langs:
            detected_langs = ["en"]
        
        # Basic sentiment analysis
        tone_data = []
        sentences = [s.strip() for s in raw_text.split('.') if s.strip()]
        segment_size = max(1, len(sentences) // 8)
        segments = [sentences[i:i+segment_size] for i in range(0, len(sentences), segment_size)]
        
        for i, segment in enumerate(segments[:10]):
            segment_text = ' '.join(segment).lower()
            
            # Simple sentiment scoring
            positive_words = ['thank', 'great', 'good', 'yes', 'perfect', 'excellent']
            negative_words = ['problem', 'issue', 'no', 'not', 'angry', 'frustrated']
            
            pos_count = sum(1 for word in positive_words if word in segment_text)
            neg_count = sum(1 for word in negative_words if word in segment_text)
            
            if pos_count > neg_count:
                cust_sent = 3.5 + min(1.5, pos_count * 0.3)
                agent_energy = 3.8
            elif neg_count > pos_count:
                cust_sent = 3.0 - min(1.0, neg_count * 0.2)
                agent_energy = 4.0
            else:
                cust_sent = 3.0
                agent_energy = 3.5
            
            phase = "Opening" if i < 2 else ("Closing" if i > 6 else "Discussion")
            
            tone_data.append({
                "phase": phase,
                "agent_energy": agent_energy,
                "customer_sentiment": cust_sent
            })
        
        logger.info("✅ Transcription completed")
        return raw_text, detected_langs, tone_data
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        traceback.print_exc()
        return None, [], []


def polish_transcript(raw_transcript: str, kb_context: str) -> str:
    """Polish the transcript using KB context"""
    try:
        if not OPENAI_CLIENT or not raw_transcript:
            return raw_transcript
        
        # FIX: Truncate transcript if too long to fit within token limits
        max_transcript_chars = 8000  # Conservative limit
        if len(raw_transcript) > max_transcript_chars:
            logger.warning(f"Transcript too long ({len(raw_transcript)} chars), truncating to {max_transcript_chars}")
            raw_transcript = raw_transcript[:max_transcript_chars] + "\n[...truncated...]"
        
        prompt = f"""Polish this customer service call transcript for readability while preserving company-specific terminology.

KNOWLEDGE BASE for correct terminology:
{kb_context[:1500]}

INSTRUCTIONS:
1. Correct spelling, grammar, and punctuation
2. Maintain speaker labels
3. Preserve exact company names (especially "Wint Wealth")
4. Translate to English if needed
5. Return ONLY the polished transcript

RAW TRANSCRIPT:
{raw_transcript}
"""
        
        # FIX: Reduced max_tokens from 8000 to 4000 (within GPT-4-turbo limit)
        response = OPENAI_CLIENT.chat.completions.create(
            model=ANALYSIS_MODEL,
            max_tokens=4000,  # FIXED: Was 8000, now 4000
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        
        polished = response.choices[0].message.content.strip()
        logger.info("✅ Transcript polished")
        return polished
        
    except Exception as e:
        logger.error(f"Polishing error: {e}")
        return raw_transcript


def analyze_call(transcript: str, kb_context: str) -> Tuple[Dict[str, Any], float, str]:
    """Deep analysis of the call"""
    try:
        if not OPENAI_CLIENT:
            raise Exception("OpenAI client not initialized")
        
        prompt = f"""You are a QA analyst. Analyze this call transcript and provide scoring based on the Knowledge Base.

TRANSCRIPT:
{transcript}

KNOWLEDGE BASE:
{kb_context}

Return ONLY a valid JSON object with this exact structure:
{{
  "call_opening": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "call_closing": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "technically_legally_correct": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "all_questions_addressed": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "expectation_setting": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "process_adherence": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "vocabulary_grammar_pronunciation": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "fillers_fumbling_clarity": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "energy_tone_modulation": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "active_listening_interruptions": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "simplifying_answers": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "empathy": {{"score": [1-5], "justification": "...", "transcript_quote": "...", "kb_recommended_answer": "..."}},
  "unanswered_questions": [],
  "emotional_arc": "...",
  "overall_tone": "..."
}}

SCORING: 5=Perfect, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable
"""
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = OPENAI_CLIENT.chat.completions.create(
                    model=ANALYSIS_MODEL,
                    max_tokens=4096,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                raw_response = response.choices[0].message.content.strip()
                
                # Clean and parse JSON
                json_str = raw_response
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]
                
                evaluation = json.loads(json_str.strip())
                
                # Calculate total score
                total_score = 0
                for key, weight in RUBRIC.items():
                    raw_score = float(evaluation.get(key, {}).get("score", 3))
                    weighted_score = ((raw_score - 1) / 4) * weight
                    total_score += weighted_score
                
                total_score = round(total_score, 2)
                
                # Performance level
                percentage = (total_score / MAX_SCORE) * 100
                if percentage >= 90:
                    level = "Outstanding"
                elif percentage >= 80:
                    level = "Exceeds Expectations"
                elif percentage >= 70:
                    level = "Meets Expectations"
                elif percentage >= 60:
                    level = "Needs Improvement"
                else:
                    level = "Unsatisfactory"
                
                logger.info(f"✅ Analysis completed: {total_score}/{MAX_SCORE} ({level})")
                return evaluation, total_score, level
                
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt+1} JSON parse failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2)
        
        raise Exception("All analysis attempts failed")
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        traceback.print_exc()
        raise


def log_to_google_sheet(call_data: Dict[str, Any], analysis_data: Dict[str, Any]):
    """Log comprehensive call data and analysis to Google Sheet"""
    try:
        creds_dict = json.loads(GSPREAD_CREDENTIALS)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        # FIX: Better error handling for sheet access - Open by ID
        try:
            sheet = client.open_by_key(SHEET_ID).sheet1
            
            # Ensure headers exist
            ensure_sheet_headers(sheet)
            
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"❌ Spreadsheet with ID '{SHEET_ID}' not found. Please check sharing permissions.")
            logger.error(f"   Make sure the service account email has Editor access to the sheet.")
            raise
        
        # Get current local time for analysis timestamp
        current_time = get_local_time()
        
        # Format Exotel timestamps to IST
        def convert_to_ist(exotel_timestamp):
            """Convert Exotel timestamp to IST with proper formatting"""
            if not exotel_timestamp or exotel_timestamp == 'Unknown':
                return 'N/A'
            
            try:
                dt = datetime.strptime(str(exotel_timestamp).strip(), '%Y-%m-%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S') + f' {TIMEZONE_NAME}'
            except Exception as e:
                logger.warning(f"Timestamp conversion failed for {exotel_timestamp}: {e}")
                return str(exotel_timestamp) + f' {TIMEZONE_NAME}'
        
        # Prepare comprehensive row data with IST timestamps
        row = [
            format_ist_timestamp(current_time),
            call_data.get('Sid', ''),
            convert_to_ist(call_data.get('DateCreated', '')),
            convert_to_ist(call_data.get('StartTime', '')),
            convert_to_ist(call_data.get('EndTime', '')),
            call_data.get('Duration', ''),
            call_data.get('From', {}).get('PhoneNumber', '') if isinstance(call_data.get('From'), dict) else call_data.get('From', ''),
            call_data.get('To', {}).get('PhoneNumber', '') if isinstance(call_data.get('To'), dict) else call_data.get('To', ''),
            call_data.get('Status', ''),
            call_data.get('Direction', ''),
            call_data.get('AnsweredBy', ''),
            call_data.get('Price', ''),
            call_data.get('RecordingUrl', ''),
            analysis_data.get('total_score', ''),
            f"{(analysis_data.get('total_score', 0) / MAX_SCORE * 100):.1f}%",
            analysis_data.get('performance_level', ''),
            analysis_data.get('conversation_tone', ''),
            ', '.join(analysis_data.get('detected_languages', [])),
            ', '.join(analysis_data.get('unanswered_questions', [])),
            analysis_data.get('emotional_arc', ''),
        ]
        
        # Add detailed scores for each rubric parameter
        evaluation = analysis_data.get('evaluation', {})
        for key in RUBRIC.keys():
            param_data = evaluation.get(key, {})
            row.extend([
                f"{param_data.get('score', 'N/A')}/5",
                param_data.get('justification', 'N/A'),
                param_data.get('transcript_quote', 'N/A'),
                param_data.get('kb_recommended_answer', 'N/A')
            ])
        
        # Add transcript at the end
        row.append(analysis_data.get('transcript', ''))
        
        # FIX: Add retry logic and better error handling
        max_retries = 3
        for retry in range(max_retries):
            try:
                sheet.append_row(row, value_input_option='USER_ENTERED')
                logger.info(f"✅ Logged to Google Sheet: {call_data.get('Sid', 'Unknown')}")
                return
            except Exception as append_error:
                logger.warning(f"Attempt {retry+1} to append row failed: {append_error}")
                if retry == max_retries - 1:
                    raise
                time.sleep(2)
        
    except Exception as e:
        logger.error(f"❌ Error logging to sheet: {e}")
        logger.error(f"Sheet ID: {SHEET_ID}")
        logger.error(f"Call SID: {call_data.get('Sid', 'Unknown')}")
        traceback.print_exc()
        raise  # Re-raise to mark call as failed


def process_call(call: Dict[str, Any], kb_context: str) -> bool:
    """Process a single call: transcribe, analyze, and log"""
    try:
        call_sid = call.get('Sid', 'Unknown')
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Call SID: {call_sid}")
        logger.info(f"{'='*60}")
        
        # Get recording URL
        recording_url = call.get('PreSignedRecordingUrl') or call.get('RecordingUrl')
        
        if not recording_url:
            logger.warning(f"No recording URL for call {call_sid}")
            return False
        
        # Step 1: Transcribe
        logger.info("Step 1/4: Transcribing audio...")
        raw_transcript, languages, tone_data = transcribe_audio(recording_url)
        
        if not raw_transcript:
            logger.error(f"Transcription failed for {call_sid}")
            return False
        
        # Step 2: Polish
        logger.info("Step 2/4: Polishing transcript...")
        polished_transcript = polish_transcript(raw_transcript, kb_context)
        
        # Step 3: Analyze
        logger.info("Step 3/4: Analyzing call quality...")
        evaluation, total_score, performance_level = analyze_call(polished_transcript, kb_context)
        
        # Prepare comprehensive analysis data
        analysis_data = {
            'transcript': polished_transcript,
            'evaluation': evaluation,
            'total_score': total_score,
            'performance_level': performance_level,
            'conversation_tone': evaluation.get('overall_tone', 'N/A'),
            'detected_languages': languages,
            'unanswered_questions': evaluation.get('unanswered_questions', []),
            'emotional_arc': evaluation.get('emotional_arc', 'N/A'),
            'tone_data': tone_data
        }
        
        # Step 4: Log to Google Sheet
        logger.info("Step 4/4: Logging to Google Sheet...")
        log_to_google_sheet(call, analysis_data)
        
        logger.info(f"✅ Successfully processed call {call_sid}")
        logger.info(f"   Score: {total_score}/{MAX_SCORE} ({performance_level})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing call {call.get('Sid', 'Unknown')}: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function"""
    logger.info("\n" + "="*80)
    logger.info("🚀 AUTOMATED QA ENGINE STARTED")
    logger.info("="*80 + "\n")
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)
    
    # Load KB context
    logger.info("📚 Loading Knowledge Base...")
    kb_context = get_kb_context()
    logger.info(f"✅ KB loaded: {len(kb_context)} characters")
    
    # FIX: Get already processed calls from sheet
    logger.info("\n📋 Checking already processed calls...")
    try:
        creds_dict = json.loads(GSPREAD_CREDENTIALS)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
        
        # Ensure headers are set up
        ensure_sheet_headers(sheet)
        
        # Get already processed calls
        processed_sids = get_already_processed_calls(sheet)
    except Exception as e:
        logger.warning(f"Could not access sheet to check processed calls: {e}")
        processed_sids = set()
    
    # Fetch recordings from last hour
    logger.info("\n📞 Fetching recordings from last hour...")
    calls = fetch_last_hour_recordings()
    
    if not calls:
        logger.info("ℹ️  No recordings found in the last hour")
        return
    
    # FIX: Filter out already processed calls
    unprocessed_calls = [call for call in calls if call.get('Sid') not in processed_sids]
    
    logger.info(f"\n📊 Total recordings found: {len(calls)}")
    logger.info(f"📊 Already processed: {len(calls) - len(unprocessed_calls)}")
    logger.info(f"📊 New recordings to process: {len(unprocessed_calls)}\n")
    
    if not unprocessed_calls:
        logger.info("✅ All calls already processed. Nothing to do.")
        return
    
    # Process each call
    successful = 0
    failed = 0
    
    for i, call in enumerate(unprocessed_calls, 1):
        logger.info(f"\n--- Processing Call {i}/{len(unprocessed_calls)} ---")
        
        if process_call(call, kb_context):
            successful += 1
        else:
            failed += 1
        
        # Rate limiting: wait between calls
        if i < len(unprocessed_calls):
            logger.info("⏳ Waiting 10 seconds before next call...")
            time.sleep(10)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 EXECUTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total New Calls Found: {len(unprocessed_calls)}")
    logger.info(f"✅ Successfully Processed: {successful}")
    logger.info(f"❌ Failed: {failed}")
    if unprocessed_calls:
        logger.info(f"Success Rate: {(successful/len(unprocessed_calls)*100):.1f}%")
    logger.info("="*80 + "\n")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
