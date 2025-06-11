from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response, LaunchRequest, IntentRequest, SessionEndedRequest
from ask_sdk_core.utils import is_intent_name
import pymongo
import os
import logging
from datetime import datetime
from datetime import datetime
from zoneinfo import ZoneInfo
import google.generativeai as genai
import json
import random
import re
from typing import Optional

# Configure logging for debugging and tracking errors
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# MongoDB connection setup 
MONGO_URI = os.getenv("MONGO_URI")  # Retrieve MongoDB connection string from environment variables
if not MONGO_URI:
    raise ValueError("MongoDB connection string is missing. Set MONGO_URI in Lambda environment variables.")

def get_db():
    """Returns a single MongoDB connection to be reused across functions"""
    if not hasattr(get_db, "client"):
        get_db.client = pymongo.MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=False) # Connect to MongoDB
    return get_db.client['test'] # Return the database

db = get_db()
collection = db['questions']


# Initialise SkillBuilder (Manages Alexa skill handlers)
sb = SkillBuilder()

# Error message when data retrieval fails
retrieve_error_mg = "Sorry, I couldn't retrieve the questions. Please try again later."

# Normalise question format
def normalise_question(text):
    # Replace e.g., e.g: or e.g. with "for example"
    text = re.sub(r"\be\.g[\.:]?\s*", "for example, ", text, flags=re.IGNORECASE)
    # Remove brackets ()
    text = re.sub(r"[()]", "", text)
    return text.strip()

# Retrieve questions from MongoDB
def get_questions() -> dict | None:
    """
    Fetches the first document from the 'questions' collection in MongoDB.

    Returns:
        dict: The questions document without the MongoDB '_id' field.
        None: If no document is found or an error occurs.
    """
    try:
        questions_doc = collection.find_one()  # Get the first document
        if not questions_doc:
            logger.info("No document found in 'questions' collection.")
            return None

        # Remove MongoDB ObjectId before returning
        questions_doc.pop("_id", None)

        # Normalise question texts
        if "sections" in questions_doc:
            for section in questions_doc["sections"]:
                for q in section.get("questions", []):
                    q["question"] = normalise_question(q["question"])

        logger.info(f"Retrieved questions: {json.dumps(questions_doc)}")
        return questions_doc

    except Exception as e:
        logger.error(f"Error retrieving questions from MongoDB: {str(e)}")
        return None

# Retrieve the next question based on the current section and question index
def get_next_question(questions, section_index, question_index):
    """Fetches the next question from the MongoDB document"""
    if questions and "sections" in questions:
        sections = questions['sections']
        if section_index < len(sections):
            current_section = sections[section_index]
            if question_index < len(current_section['questions']):
                return current_section['questions'][question_index]  # Returns full question object
    return None  # No more questions in this section

# Generates auto-incrementing IDs for patients and sessions
def get_next_sequence(name):
    """Auto-increments the session_id or patient_id in the 'counters' collection"""
    counters_collection = db["counters"]
    counter = counters_collection.find_one_and_update(
        {"_id": name}, 
        {"$inc": {"seq": 1}}, 
        upsert=True,  
        return_document=pymongo.ReturnDocument.AFTER  
    )
    return counter["seq"]

# Retrieve session attributes
def get_session_attributes(handler_input):
    """Returns the current session attributes"""
    return handler_input.attributes_manager.session_attributes

# Store patient data into MongoDB
def save_patient_data(session_attributes):
    """Saves or updates a question-response pair for a given patient session in MongoDB."""
    # Connect to the MongoDB collection
    patient_collection = db['patients']  # Connect to the 'patients' collection in MongoDB
    # Get the current time in Melbourne timezone
    melbourne_time = datetime.now(ZoneInfo("Australia/Melbourne"))

    # Extract session_id and patient_id from session attributes
    session_id = session_attributes.get("session_id")
    patient_id = session_attributes.get("patient_id")

    # Loop through each question-response pair stored in session attributes
    for question_id, (question_text, answer) in session_attributes.get("patient_data", {}).items():
        
        # Define a query to check if the specific question_id already exists in the response array
        update_query = {
            "session_id": session_id,
            "patient_id": patient_id,
            "response.question_id": question_id  # Search for this specific question_id in the array
        }

        # Define the update operation to modify an existing response
        update_action = {
            "$set": {
                "session_info.session_start": session_attributes.get("session_start"),  # Update session start time
                "session_info.session_end": melbourne_time,  # Set session end time to current time
                
                # Update the existing response where question_id matches
                "response.$.question": question_text,  # Update the question text if needed
                "response.$.response": answer,  # Update the stored response
                "response.$.time": melbourne_time  # Update the response timestamp
            }
        }

        # Try to update an existing response in MongoDB
        result = patient_collection.update_one(update_query, update_action)

        # If no existing response was found, insert a new one
        if result.matched_count == 0:
            patient_collection.update_one(
                {"session_id": session_id, "patient_id": patient_id},  # Find the correct patient session
                {
                    "$push": {  # Add a new response entry
                        "response": {
                            "question_id": question_id,  # Store question_id (e.g., "q0_0")
                            "question": question_text,  # Store slot_name or the question text
                            "response": answer,  # Store the user's validated answer
                            "time": melbourne_time  # Store the timestamp
                        }
                    }
                },
                upsert=True  # Create a new document if it doesn’t exist
            )

    # Log a success message indicating that data has been saved/updated
    logger.info(f"Patient data saved/updated successfully for session {session_id}.")


# Set up Gemini API Key
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("API Key is missing. Please ensure the GENAI_API_KEY environment variable is set.")

# Configure Gemini API with the retrieved API key
genai.configure(api_key=GENAI_API_KEY)

def get_gemini_response(user_answer, question):
    """
    Sends user input and the previous question to Gemini for a dynamic follow-up response.
    Returns the AI-generated follow-up question.
    """
    
    # Define the model to use
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    
    # Define the prompt for generating a follow-up question
    prompt = f"Patient was asked: '{question}'\nPatient responded: '{user_answer}'\nWhat is the best follow-up question to ask next? Return only the follow-up question."

    try:
        response = model.generate_content(prompt)

        # Extract AI-generated response
        if hasattr(response, 'text') and response.text:
            return response.text.strip()

        return None  # If no text, return None (fallback to predefined questions)

    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
        return None  # If Gemini API fails, use fallback

# Check if the user response is a YES, NO, or UNCLEAR
def check_yes_no_with_gemini(user_response: str, question: str) -> str:
    """Returns YES, NO, or UNCLEAR."""
    prompt = f"""
    You are a **medical voice assistant** helping with patient intake.

    **Persona:** 
    - You are a careful and attentive voice assistant supporting healthcare staff.
    - Your job is to understand whether a patient's reply indicates agreement, disagreement, or uncertainty.

    **Context:** 
    - A patient was asked a **confirmation question** to verify their information.
    - They responded verbally. 
    - Their answer might be short ("yes", "no") or longer ("yes, that's correct", "no, that's not my name").

    **Task:** 
    - Based only on the patient's verbal response, determine whether they confirmed (YES), denied (NO), or gave an unclear response (UNCLEAR).

    **Patient Confirmation Question:** 
    "{question}"

    **Patient Response:** 
    "{user_response}"

    **Output Format:** 
    Return exactly one word:
    - YES
    - NO
    - UNCLEAR

    **Constraints:** 
    - Short answers like "yes" or "no" are completely acceptable.
    - Accept informal affirmatives like "yes", "it is", "that's right", "correct", "sure", "absolutely", "yeah", "yep", and similar.
    - Also accept "yes I can", "yes I do", "yes I am" "yes unfortunately", and similar phrasing as YES.
    - Accept informal negatives like "no", "it's not", "that's wrong", "not really", "nope", "nah", and similar.
    - A response like "Give me a second", "Hold on", "I need a minite" to "Are you ready?" question should be considered as "NO".
    - Do not explain, do not rephrase, do not apologise.
    - Only output YES, NO, or UNCLEAR.
    - Do not add punctuation or extra text.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        decision = result.text.strip().upper()
        logger.info(f"[Gemini Confirmation] Detected response: {decision}")
        return decision
    except Exception as e:
        logger.error(f"[Gemini Confirmation] Error calling Gemini: {str(e)}")
        return "UNCLEAR"

def parse_yes_no_and_detail(user_response: str, main_question: str, topic: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Analyses the patient's response using Gemini to:
    1. Determine if it implies YES, NO, or UNCLEAR to the main Yes/No question.
    2. Extract follow-up detail if the answer is YES and contains specifics.
    
    Returns:
        Tuple[str, Optional[str]]:
            - One of "YES", "NO", "UNCLEAR"
            - Extracted detail (e.g., "asthma and high blood pressure") or None
    """
    prompt = f"""
    You are a reliable and medically informed assistant designed to interpret patient responses during health intake.

    **Main Question:** "{main_question}"
    **Patient Response:** "{user_response}"

    Your tasks are:
    1. Determine if the patient is affirming ("YES"), denying ("NO"), or unclear ("UNCLEAR") in response to the main question.
    2. If the answer is "YES", and the patient provides additional detail, extract that detail clearly, concisely and medically — even if it's embedded in casual language.

    Examples:
    - "yeah I have asthma and high blood pressure" → YES|asthma and high blood pressure
    - "yes I had my appendix removed" → YES|appendectomy
    - "I had hip replacement and some dental work" → YES|hip replacement
    - "nope" → NO|
    - "not sure" → UNCLEAR|

    Rules:
    - Return the result in one line only: DECISION|DETAIL
    - If there is no detail to extract, leave the detail section blank. E.g., "YES|", "NO|", or "UNCLEAR|"
    - If the detail implies a known medical procedure, summarise it in a medically accurate form (e.g., "had appendix out" → "appendectomy").
    - Do NOT include any explanation, punctuation outside the format, or markdown
    - Do NOT include line breaks, bullet points, or elaboration — only the result

    Format:
    DECISION|[EXTRACTED_DETAIL]
    """.strip()

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        response_text = result.text.strip()

        logger.info(f"[Gemini parse_yes_no_and_detail] {response_text}")

        if "|" in response_text:
            decision, detail = response_text.split("|", 1)
            return decision.strip().upper(), detail.strip() or None
        return "UNCLEAR", None
    except Exception as e:
        logger.error(f"parse_yes_no_and_detail failed: {e}")
        return "UNCLEAR", None


# Rephrase the question using Gemini AI
def is_repeat_request(user_text: str, original_question: str) -> bool:
    prompt = f"""
    You are an attentive and patient digital assistant working in a medical clinic. 
    Your role is to detect when a patient is confused or didn't understand a question during a voice-based medical interview.
    Patients are asked clear, structured questions, and their responses are recorded as natural spoken text. 
    Your job is to assess whether the response indicates understanding or confusion.

    The patient was asked: "{original_question}"
    The patient responded: "{user_text}"

    Determine if the patient:
    - Understood the question and responded appropriately (return NO)
    - Misunderstood, was unsure, or asked for clarification (return YES)

    Return exactly one of the following values:
    - YES — if the patient showed signs of confusion, hesitation, or unrelated responses
    - NO — if the patient clearly answered the question

    Only return YES or NO.
    Do not provide explanations or summaries.

    Examples of confusion:
    - The patient says things like "sorry", "what", "not sure", "I don't know", or "can you repeat".
    - The answer is off-topic or unrelated.
    - The response is too vague to be useful.

    Examples of understanding:
    - The patient provides a relevant answer to the question, even if it's informal or not perfectly formatted.
    - For name questions, responses like "John Smith", "My name is Sarah Lee", or "I'm Tom" are valid.

    Output format: 
    Just one word, either YES or NO.
    Do not include any explanation.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        decision = result.text.strip().lower()
        logger.info(f"Gemini repeat check result: {decision}")
        return decision.startswith("yes")
    except Exception as e:
        logger.warning(f"Gemini repeat detection failed: {e}")
        return False


# Validate user input using Gemini AI
def validate_with_gemini(slot_name: str, slot_value: str, original_question: str) -> tuple[bool, str]:
    """
    Uses Gemini to validate structured and unstructured patient responses.
    Returns:
      - (True, value) if valid
      - (False, reworded_question) if invalid or unclear
    """

    def generate_prompt(slot_name: str, slot_value: str, original_question: str, validation_rule: str) -> str:
        return f"""
        You are a smart and meticulous digital assistant supporting a voice-based medical intake system.
        You assist clinicians by checking whether a patient's spoken response matches the expected information type and format.

        The patient was asked: "{original_question}"
        Patient Response: "{slot_value}"
        Slot type: "{slot_name}"

        Validation Rule:
        {validation_rule}

        Task:
        Determine whether the patient's response satisfies the expected intent of the question.

        Output Format:
        - VALID — if the response clearly answers the question
        - VALID|[Formatted Value] — if the response is valid but needs formatting
        - INVALID|[Reworded Question] — if the response is vague, unclear, or incorrect

        Constraints:
        - Do not include explanations or multiple options.
        - Format expected values as follows:
        - Dates: YYYY-MM-DD
        - Phone numbers: +61 format
        - Emails: standard email format (e.g. someone@example.com)
        - Addresses: must include street number, street name, suburb and state
        - Names: at least two alphabetic words, capitalised
        - Gender: Accept synonyms of 'male' or 'female' such as woman, man, etc. and convert it to female or male
        - Relationships: Accept common relationships like mother, father, sister, etc.

        Only return one of: VALID, VALID|[Formatted Value], or INVALID|[Reworded Question]
                """.strip()

    # Structured validation rules
    name_rule = "Ensure the response contains at least two alphabetic words representing a typical full name (e.g., 'John Smith')."
    phone_rule = "Convert spoken input to digits. Accept valid Australian numbers with +61 prefix if starting with 0."
    date_rule = "Accept full or partial dates. If the patient says 'March twenty twenty three' or Alexa transcribes it as 'March 20 '23', treat this as 'March 2023' unless a specific day is clearly mentioned."

    validation_rules = {
        "name": name_rule,
        "emergency_contact": name_rule,
        "date_of_birth": "Convert spoken input into YYYY-MM-DD format. Accept 'eleven november nineteen ninety' as 1990-11-11. Must be a past date.",
        "email": "Ensure it is a valid email. Accept spoken versions like 'john dot doe at gmail dot com'.",
        "gender": "Accept only 'male' or 'female' or common synonyms like 'man', 'woman'.",
        "contact_number": phone_rule,
        "emergency_contact": name_rule,
        "emergency_contact_phone": phone_rule,
        "home_address": "Must include street number, street name, suburb, state. Post code is optional. Convert spoken numbers/phrases into standard address format.",
        "emergency_contact_relationship": "Accept typical relationships (e.g., mother, father, partner, friend, spouse).",
        "surgeries_date": date_rule,
        "immunizations_date": date_rule
    }

    # Use specific or fallback rule
    rule = validation_rules.get(slot_name, f"""
    Ensure the patient's response clearly answers the question: '{original_question}'.
    - It should be complete and relevant.
    - If unclear or incomplete, request clarification.
    """)

    prompt = generate_prompt(slot_name, slot_value, original_question, rule)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        response_text = result.text.strip()

        logger.info(f"Gemini validation response: {response_text}")

        if response_text.startswith("VALID|"):
            return True, response_text.split("|", 1)[1].strip("[]")

        elif response_text.startswith("INVALID|"):
            return False, response_text.split("|", 1)[1].strip("[]")

        elif response_text == "VALID":
            return True, slot_value

        return True, slot_value  # Fallback if unexpected

    except Exception as e:
        logger.error(f"Gemini validation failed: {e}")
        return True, slot_value  # Fallback if Gemini fails


# Extract structured information from free-text responses using Gemini AI
def extract_information_with_gemini(question: str, user_response: str, slot_name: Optional[str] = None) -> str:
    """
    Uses Gemini AI to extract structured details from free-text answers.
    Customises prompts based on slot_name when available.
    """
    if slot_name == "emergency_contact_relationship":
        prompt = f"""
        The patient was asked to provide the relationship of their emergency contact.
        They responded: "{user_response}"

        Extract only the relationship term such as "father", "sister", "partner", etc. 
        Return just the single word, with no extra explanation.
        """
    else:
        prompt = f"""
        The patient was asked: "{question}"
        The patient responded: "{user_response}"

        Extract the key medical details in a concise format (e.g., "hypertension, diabetes").
        If nothing can be extracted, return the original response.
        """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        return result.text.strip() if result.text else user_response
    except Exception as e:
        logger.error(f"Error extracting structured response with Gemini: {str(e)}")
        return user_response

    
# Rephrase the question using Gemini
def get_rephrased_question(original_question: str) -> str:
    prompt = f"""
    The following is a question we ask patients during a medical interview: "{original_question}"

    Rephrase this question to be simpler and easier to understand for the average patient.

    Important instructions:
    - Return **only one** rephrased version in a conversatinal format.
    - Do **not** return multiple options, bullet points, markdown, or explanations.
    - Do **not** include any introductory text like "Here’s a simpler version".
    - Return the rephrased question as a single plain sentence.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        full_text = result.text.strip() if result.text else None

        if not full_text:
            return None

        # If Gemini returned a list, extract the first item
        for line in full_text.splitlines():
            line = line.strip()
            if line.startswith("**") or line.startswith("1.") or line.startswith("-"):
                return re.sub(r"^[\*\-\d\.]+", "", line).strip(' "')

        # Otherwise, just return the full response
        return full_text

    except Exception as e:
        logger.warning(f"Gemini failed to rephrase question: {e}")
        return None

# Determine which follow-up questions can be skipped based on the patient's response
def get_skipped_followups(main_question: str, patient_response: str, follow_ups: list) -> list:
    """
    Use Gemini to determine which follow-up questions are already answered in the initial response.
    Returns a list of indexes (0-based) of follow-ups to skip.
    """
    prompt = f"""
    The patient was asked: "{main_question}"
    They replied: "{patient_response}"

    These are the planned follow-up questions:
    {chr(10).join([f"{i+1}. {q['question']}" for i, q in enumerate(follow_ups)])}

    Based on their reply, which follow-up questions are already answered?

    Return only the question numbers to skip, as a comma-separated list (e.g., 1,3). If none, return "none".
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        result = model.generate_content(prompt)
        text = result.text.strip().lower()
        if "none" in text:
            return []
        # Convert string like "1,3" to [0,2]
        return [int(i.strip()) - 1 for i in text.split(",") if i.strip().isdigit()]
    except Exception as e:
        logger.warning(f"[Gemini] Failed to analyse follow-up skipping: {e}")
        return []


# Handles the session lifecycle - Add short pauses to the text
def add_short_pause(text, pause_duration_ms=1000):
    return f"<break time='{pause_duration_ms}ms'/> {text}"


# Converts "2023-03" into "March 2023" for natural Alexa speech
def format_date_for_speech(ym: str) -> str:
    """Convert '2023-03' to 'March 2023' for natural speech."""
    try:
        dt = datetime.strptime(ym, "%Y-%m")
        return dt.strftime("%B %Y")  # e.g., "March 2023"
    except Exception:
        return ym  # fallback to raw if parsing fails


# Handles session end
class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput):
        return isinstance(handler_input.request_envelope.request, SessionEndedRequest)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Session ended.")
        return handler_input.response_builder.response

sb.add_request_handler(SessionEndedRequestHandler())


# Log full request payload inside the handler
def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to a string format
    raise TypeError(f"Type {type(obj)} not serializable")


# LaunchRequest handler
@sb.request_handler(can_handle_func=lambda handler_input:
                    isinstance(handler_input.request_envelope.request, LaunchRequest))
def launch_request_handler(handler_input: HandlerInput) -> Response:
    """Handles the launch of the Alexa skill and starts asking questions"""
    questions = get_questions()
    session_attributes = get_session_attributes(handler_input)
    
    # Get the current time in Melbourne timezone
    melbourne_time = datetime.now(ZoneInfo("Australia/Melbourne"))

    if questions:
        session_attributes.update({
            'current_section': 0,
            'current_question': 0,
            'session_start': melbourne_time,
            'session_id': get_next_sequence("session_id"),
            'patient_id': get_next_sequence("patient_id")
        })
        speak_output = questions['opening']
        session_attributes['waiting_for_ready_confirmation'] = True

    else:
        speak_output = retrieve_error_mg

    return handler_input.response_builder.speak(speak_output).ask(speak_output).response


# Registers this function as the handler for CaptureAnswerIntent
@sb.request_handler(can_handle_func=lambda handler_input:
    isinstance(handler_input.request_envelope.request, IntentRequest) and
    handler_input.request_envelope.request.intent.name == "CaptureAnswerIntent")   

# CaptureAnswerIntent handler
def capture_answer_intent(handler_input: HandlerInput) -> Response:
    logger.info("CaptureAnswerIntent matched successfully.")    # Logs that CaptureAnswerIntent has been successfully triggered
    session_attributes = handler_input.attributes_manager.session_attributes    # Fetches session attributes (memory between turns)

    # List of free text slots
    FREE_TEXT_SLOTS = [
        "medical_conditions_list",
        "surgeries_list",
        "surgeries_reason",
        "medications_list",
        "allergy_type",
        "allergy_reaction",
        "family_medical_conditions_list",
        "family_medical_conditions_list_2",
        "family_disability_type",
        "family_disability_type_2",
        "exercise_list",
        "immunizations_list",
        "occupation",
        "travel_frequency",
        "harsh_environment_exposure"
    ]

    # List of slots that do NOT need confirmation (e.g., Yes/No or system-handled data)
    NO_CONFIRMATION_SLOTS = [
       "medical_conditions", "surgeries", "allergies", "children",
        "smoking_status", "alcohol_consumption", "exercise", "family_medical_history", "family_disabilities",
        "immunizations", "previous_medical_records", "previous_medical_records_submission"
    ]

    # Try to get saved questions
    questions = session_attributes.get('questions')
    # If missing, load questions from database or backend via get_questions()
    if not questions:
        questions = get_questions()
        # If still no questions (e.g., error retrieving), return an error message asking the user to try again.
        if not questions:
            return handler_input.response_builder.speak(retrieve_error_mg).ask(retrieve_error_mg).response
        
        # If questions are loaded successfully, get the current time in Melbourne timezone
        melbourne_time = datetime.now(ZoneInfo("Australia/Melbourne"))
        # If loading questions succeeded, set up session tracking: section, question, and session start time.
        session_attributes.update({
            'questions': questions,
            'current_section': 0,
            'current_question': 0,
            'session_start': melbourne_time,
            'session_id': get_next_sequence("session_id"),
            'patient_id': get_next_sequence("patient_id"),
            'patient_data': {}
        })
    # Load the current section number and question number from session
    current_section = session_attributes.get('current_section', 0)
    current_question = session_attributes.get('current_question', 0)
    # Fetch the actual current question data using a helper get_next_question()
    current_question_data = get_next_question(questions, current_section, current_question)
    # Override slot_name and question_text if we're in a follow-up
    if session_attributes.get("current_followup_for") and not session_attributes.get("awaiting_followup_confirmation"):
        unconfirmed = session_attributes.get("unconfirmed_followup", {})
        if isinstance(unconfirmed, dict) and all(k in unconfirmed for k in ["question_title", "question_text"]):
            slot_name = unconfirmed["question_title"]
            question_text = unconfirmed["question_text"]
        else:
            slot_name = current_question_data.get("question_title", "response")
            question_text = current_question_data.get("question", "")
    else:
        slot_name = current_question_data.get("question_title", "response")
        question_text = current_question_data.get("question", "")

    # If no question is found, it means survey is over — say thank you and finish
    if not current_question_data:
        return handler_input.response_builder.speak("No further questions. Thank you.").response

    # Get the intent object
    intent_request = handler_input.request_envelope.request.intent
    # Figure out what slot to check (question_title) or default to "response."
    slot_name = current_question_data.get("question_title", "response")
    # Ensure slot_name/question_text comes from follow-up if in follow-up phase
    if session_attributes.get("current_followup_for") and not session_attributes.get("awaiting_followup_confirmation"):
        unconfirmed = session_attributes.get("unconfirmed_followup", {})
        if isinstance(unconfirmed, dict):
            slot_name = unconfirmed.get("question_title", slot_name)
            question_text = unconfirmed.get("question_text", question_text)
    # Try to extract a non-empty slot value from the request
    slot_value = next((s.value for s in intent_request.slots.values() if s and s.value), None)

    # If no slot value is found directly, double-check all slots just in case
    if not slot_value:
        for key, alt_slot in intent_request.slots.items():
            if alt_slot and alt_slot.value:
                slot_value = alt_slot.value
                break

    # If still nothing, ask the user to repeat            
    if not slot_value:
        return handler_input.response_builder.speak(f"Sorry, could you please repeat your {slot_name.replace('_', ' ')}?").ask(f"Can you repeat your {slot_name.replace('_', ' ')}?").response

    # Log the slot name and value for debugging
    logger.info(f"[DEBUG] Slot name: {slot_name}")
    logger.info(f"[DEBUG] Slot value: {slot_value}")

    # Readiness Confirmation Phase
    if session_attributes.get("waiting_for_ready_confirmation"):

        if "opening" not in questions:
            questions = get_questions()
            session_attributes["questions"] = questions

        # Use Gemini AI to interpret YES/NO/UNCLEAR from user's speech based on opening question
        decision = check_yes_no_with_gemini(slot_value, questions["opening"])
        # If YES -> Mark ready, reset counters, move to the first question
        if decision == "YES":
            session_attributes["waiting_for_ready_confirmation"] = False
            session_attributes["current_section"] = 0
            session_attributes["current_question"] = 0
            next_question = get_next_question(questions, 0, 0)
            speak_output = f"Great! Let's get started. {next_question['question']}"
            return handler_input.response_builder.speak(speak_output).ask(next_question["question"]).response
        # If NO -> Stay in waiting mode
        elif decision == "NO":
            return handler_input.response_builder.speak("That's okay. Let me know when you're ready.").ask("Say 'I'm ready' when you are ready.").response
        # If unclear -> rephrase and re-ask
        else:
            OPENING_REPHRASES = [
                "We’d like to ask you some simple questions to get to know your health background. Your answers are kept confidential and help our team look after you better. Shall we get started?",
                "To get to know you better and support your care, we’ll ask a few health questions. Your answers stay private. Ready to begin?",
                "Let's go through some simple health questions. Your information is kept secret and helps us provide the best care. Are you ready to start?"
            ]
            rephrased_output = random.choice(OPENING_REPHRASES)
            return handler_input.response_builder.speak(rephrased_output).ask(rephrased_output).response

    # Follow-up Confirmation Phase
    if session_attributes.get("awaiting_followup_confirmation"):
        pending = session_attributes.get("unconfirmed_followup")

        # 🪵 Debugging block
        logger.info("[DEBUG] Follow-up confirmation pending object:")
        logger.info(json.dumps(pending, indent=2, default=str))

        required_keys = ["question_id", "followup_index", "question_title", "question_text"]
        missing_keys = [key for key in required_keys if key not in pending]

        if missing_keys:
            for key in missing_keys:
                logger.warning(f"[DEBUG] Missing key in unconfirmed_followup: {key}")
            return handler_input.response_builder.speak(
                "I am sorry, but something went wrong. Could you repeat your answer?"
            ).ask("Could you please repeat that?").response

        try:
            decision = check_yes_no_with_gemini(slot_value, session_attributes["confirmation_prompt"])
        except Exception as e:
            logger.error(f"Gemini follow-up confirmation failed: {e}")
            decision = "UNCLEAR"

        if decision == "UNCLEAR":
            return handler_input.response_builder.speak(
                "Sorry, can you confirm your answer?"
            ).ask("Can you confirm?").response

        question_id = pending["question_id"]
        followup_response = pending["response"]
        slot_name = pending["question_title"]

        if decision == "YES":
            session_attributes.pop("unconfirmed_followup", None)
            session_attributes["awaiting_followup_confirmation"] = False
            session_attributes["current_followup_for"] = None  # <- Added 22/05/2025

            # Save the confirmed response
            patient_data = session_attributes.get("patient_data", {})
            followup_id = f"{question_id}_{pending['followup_index']}"
            patient_data[followup_id] = (slot_name, followup_response)
            session_attributes["patient_data"] = patient_data
            save_patient_data(session_attributes)

            # Fetch next follow-up (if any)
            followup_list = session_attributes.get(f"{question_id}_followups", [])
            next_index = pending["followup_index"] + 1

            patient_first_name = session_attributes.get("patient_first_name", "there")
            confirmation_response = add_short_pause(random.choice([
                f"Thanks {patient_first_name}! I've saved that information.",
                f"Got it! Your information has been recorded.",
                f"Thanks {patient_first_name}! Your response has been saved successfully.",
                f"Okay, {patient_first_name}. I've noted that.",
                f"Great! Your information is now stored.",
                f"Understood! Your details have been saved.",
                f"Your answer has been updated. Thank you {patient_first_name}!"
            ]), pause_duration_ms=800)

            if next_index < len(followup_list):
                next_followup = followup_list[next_index]
                session_attributes["unconfirmed_followup"] = {
                    "question_id": question_id,
                    "question_text": next_followup["question"],
                    "question_title": next_followup.get("question_title", "response"),
                    "response": "",
                    "followup_index": next_index
                }
                session_attributes["current_followup_for"] = question_id
                session_attributes["awaiting_followup_confirmation"] = False
                followup_prompt = add_short_pause(next_followup["question"], pause_duration_ms=1000)
                return handler_input.response_builder.speak(confirmation_response + followup_prompt).ask(next_followup["question"]).response

            else:
                # All follow-ups complete — move to next main question
                session_attributes["current_followup_for"] = None
                session_attributes["awaiting_followup_confirmation"] = False

                # Sync question pointer
                current_section = session_attributes.get("current_section", 0)
                session_attributes["current_question"] += 1
                current_question = session_attributes["current_question"]

                next_question = get_next_question(
                    session_attributes.get("questions"),
                    current_section,
                    current_question
                )

                if next_question:
                    speak_output = confirmation_response + add_short_pause(next_question["question"], pause_duration_ms=1000)
                    return handler_input.response_builder.speak(speak_output).ask(next_question["question"]).response

                # No more questions in this section → move to next section
                session_attributes["current_section"] += 1
                session_attributes["current_question"] = 0
                next_question = get_next_question(session_attributes.get("questions"), session_attributes["current_section"], 0)
                if next_question:
                    return handler_input.response_builder.speak(
                        confirmation_response + add_short_pause(next_question["question"], pause_duration_ms=1000)
                    ).ask(next_question["question"]).response

                return handler_input.response_builder.speak(questions.get("closing")).response


        elif decision == "NO":
            original = pending["question_text"]
            session_attributes["awaiting_followup_confirmation"] = False
            session_attributes["current_followup_for"] = None  # <- Added 22/05/2025
            session_attributes["unconfirmed_followup"] = {
                "question_id": pending["question_id"],
                "question_text": original,
                "question_title": pending.get("question_title", "response"),
                "followup_index": pending["followup_index"],
                "response": ""  # reset the answer
            }
            return handler_input.response_builder.speak(f"Okay, let's try again. {original}").ask(original).response


    # Confirmation Phase
    if session_attributes.get("awaiting_confirmation"):

        # Gemini checks if user confirmed YES/NO
        decision = check_yes_no_with_gemini(slot_value, session_attributes["confirmation_prompt"])

        # If YES -> save the pending answer as 'pending' and reset unconfirmed_answer
        if decision == "YES":
            pending = session_attributes.pop("unconfirmed_answer")
            slot_name = pending["question_title"]
            response_value = pending["response"]
            # If the question is in Section 2+, do extra information extraction
            if pending["section"] >= 2:
                try:
                    response_value = extract_information_with_gemini(pending["question_text"], response_value)
                except Exception as e:
                    logger.error(f"Error extracting info: {e}")

            # Save it in patient_data, indexed by question ID
            patient_data = session_attributes.get("patient_data", {})
            patient_data[pending["question_id"]] = (slot_name, response_value)
            session_attributes["patient_data"] = patient_data
            save_patient_data(session_attributes)

            # Check if there are follow-up questions
            follow_ups = pending.get("follow_up", [])
            # If there are follow-up questions, and patient said YES -> ask first follow-up
            if follow_ups and (
                pending["question_title"] in FREE_TEXT_SLOTS or
                (response_value.lower() in ["yes", "yeah", "yep"]) or
                any([detail is not None, session_attributes.get("unconfirmed_followup")])
            ):
                logger.info(f"Starting follow-ups for {pending['question_title']}")
                
                # Step 1: Check for pre-answered follow-ups
                skip_indexes = get_skipped_followups(pending["question_text"], response_value, follow_ups)

                # Step 2: Mark only follow-ups that should be asked
                remaining_followups = [
                    fup for i, fup in enumerate(follow_ups) if i not in skip_indexes
                ]
                # If all follow-ups are skipped, move to the next main question
                if not remaining_followups:
                    logger.info(f"All follow-ups skipped based on initial response.")
                    session_attributes["awaiting_confirmation"] = False
                    session_attributes["current_question"] += 1
                    current_section = session_attributes.gset("current_section", pending["section"])
                    current_question = pending["question_index"] + 1
                    session_attributes["current_question"] = current_question

                    # Clear leftover follow-up state from previous question
                    session_attributes.pop("unconfirmed_followup", None)
                    session_attributes["awaiting_followup_confirmation"] = False
                    session_attributes["current_followup_for"] = None

                    next_question = get_next_question(questions, current_section, current_question)

                    if not next_question:
                        session_attributes["current_section"] = current_section + 1
                        session_attributes["current_question"] = 0
                        next_question = get_next_question(questions, session_attributes["current_section"], 0)

                        if next_question:
                            return handler_input.response_builder.speak(
                                f"Thanks, {patient_first_name}. Let's move to the next section. {next_question['question']}"
                            ).ask(next_question["question"]).response
                        else:
                            return handler_input.response_builder.speak(questions.get("closing")).response

                # Step 3: Store remaining follow-ups
                session_attributes["current_followup_for"] = pending["question_id"]
                session_attributes[f"{pending['question_id']}_followups"] = remaining_followups
                first_followup = remaining_followups[0]["question"]
                return handler_input.response_builder.speak(first_followup).ask(first_followup).response

            # If no follow-up needed, move to next main question
            session_attributes["awaiting_confirmation"] = False
            session_attributes["current_question"] = pending["question_index"] + 1  # <- Important fix
            current_section = pending["section"]
            next_question = get_next_question(questions, current_section, session_attributes["current_question"])

            if next_question:
                patient_first_name = session_attributes.get("patient_first_name", "there") 
                speak_output = add_short_pause(random.choice([
                    f"Thanks {patient_first_name}! I've saved your {slot_name.replace('_', ' ')}.",
                    f"Got it! Your {slot_name.replace('_', ' ')} has been recorded.",
                    f"Thanks {patient_first_name}! Your {slot_name.replace('_', ' ')} has been saved successfully.",
                    f"Okay, {patient_first_name}. I've noted your {slot_name.replace('_', ' ')}.",
                    f"Great! Your {slot_name.replace('_', ' ')} is now stored.",
                    f"Understood! Your {slot_name.replace('_', ' ')} has been saved.",
                    f"Your {slot_name.replace('_', ' ')} has been updated. Thank you {patient_first_name}!"
                ]), pause_duration_ms=800) + add_short_pause(next_question["question"], pause_duration_ms=1000)
                return handler_input.response_builder.speak(speak_output).ask(next_question["question"]).response

            # No more questions in this section → move to next section
            session_attributes["current_section"] += 1
            session_attributes["current_question"] = 0
            next_question = get_next_question(questions, session_attributes["current_section"], 0)

            if next_question:
                patient_first_name = session_attributes.get("patient_first_name", "there") 
                return handler_input.response_builder.speak(
                    f"Thank you, {patient_first_name}. Let's move to the next section. {next_question['question']}"
                ).ask(next_question["question"]).response

            # If no more questions at all
            closing = questions.get("closing")
            return handler_input.response_builder.speak(closing).response

        elif decision == "NO":
            original = session_attributes["unconfirmed_answer"]["question_text"]
            session_attributes["awaiting_confirmation"] = False
            return handler_input.response_builder.speak(f"Okay, let's try again. {original}").ask(original).response

        else:
            return handler_input.response_builder.speak("Sorry, could you confirm?").ask("Can you confirm?").response

    # Handle Repeat Request
    repeat_check_question = (
    session_attributes.get("unconfirmed_followup", {}).get("question_text")
    if session_attributes.get("current_followup_for") and not session_attributes.get("awaiting_followup_confirmation")
    else current_question_data["question"]
    )

    if is_repeat_request(slot_value, repeat_check_question):
        rephrased = get_rephrased_question(repeat_check_question)
    
    # Detect if we are in the middle of follow-up series (not yet confirming)
    if (
        session_attributes.get("current_followup_for") and
        not session_attributes.get("awaiting_followup_confirmation")
    ):
        unconfirmed = session_attributes.get("unconfirmed_followup", {})
        
        # Safety check
        if not isinstance(unconfirmed, dict) or not all(k in unconfirmed for k in ["question_id", "followup_index", "question_title", "question_text"]):
            logger.error(f"Invalid or missing unconfirmed_followup: {unconfirmed}")
            session_attributes.pop("unconfirmed_followup", None)
            session_attributes["awaiting_followup_confirmation"] = False
            return handler_input.response_builder.speak("Sorry, something went wrong. Can you repeat that?").ask("Can you repeat that?").response

        logger.info(f"[DEBUG] (Follow-up) Overriding slot name: {slot_name}")
        logger.info(f"[DEBUG] (Follow-up) Using question text: {question_text}")

        # Try to extract structured info
        raw_response = slot_value.strip()
        normalised_response = extract_information_with_gemini(question_text, raw_response, slot_name)

        # Save and prompt for confirmation
        unconfirmed["response"] = normalised_response
        session_attributes["unconfirmed_followup"] = unconfirmed
        session_attributes["awaiting_followup_confirmation"] = True
        confirmation_prompt = f"Just to confirm, you meant: {normalised_response}. Is that correct?"
        session_attributes["confirmation_prompt"] = confirmation_prompt
        return handler_input.response_builder.speak(confirmation_prompt).ask(confirmation_prompt).response

    # Normal validation
    final_response = slot_value.strip() # Clean the answer

    if slot_name in NO_CONFIRMATION_SLOTS:
        # Use Gemini to determine YES/NO and extract detail
        decision, detail = parse_yes_no_and_detail(final_response, current_question_data["question"])
        logger.info(f"Parsed decision: {decision}, Detail: {detail}")

        # Save main question answer (Yes/No)
        patient_data = session_attributes.get("patient_data", {})
        question_id = current_question_data["question_id"]
        patient_data[question_id] = (slot_name, decision.lower())

        session_attributes["patient_data"] = patient_data
        save_patient_data(session_attributes)

        # Check for follow-ups
        follow_ups = current_question_data.get("follow_up", [])
        if not isinstance(follow_ups, list): follow_ups = []
        logger.info(f"[DEBUG] Follow-ups for {slot_name}: {follow_ups}")

        if decision == "YES":
            if detail:
                # Case 1: YES with detail → confirm that detail first
                session_attributes["current_followup_for"] = question_id
                session_attributes["unconfirmed_followup"] = {
                    "question_id": question_id,
                    "question_text": follow_ups[0]["question"],
                    "question_title": follow_ups[0].get("question_title", "response") if follow_ups else slot_name,
                    "response": detail,
                    "followup_index": 0
                }
                session_attributes["confirmation_prompt"] = f"You mentioned: {detail}. Is that correct?"
                session_attributes["awaiting_followup_confirmation"] = True
                return handler_input.response_builder.speak(session_attributes["confirmation_prompt"]).ask(session_attributes["confirmation_prompt"]).response

            elif follow_ups:
                # Case 2: YES with no detail, but follow-ups exist
                session_attributes["current_followup_for"] = question_id
                session_attributes[f"{question_id}_followups"] = follow_ups
                session_attributes["unconfirmed_followup"] = {
                    "question_id": question_id,
                    "question_text": follow_ups[0]["question"],
                    "question_title": follow_ups[0].get("question_title", "response"),
                    "response": "",
                    "followup_index": 0
                }
                session_attributes["awaiting_followup_confirmation"] = False
                return handler_input.response_builder.speak(follow_ups[0]["question"]).ask(follow_ups[0]["question"]).response
            
            else:
                logger.warning("[Fallback] YES received but no detail and no follow-ups. Skipping to next question.")

        # Move to next main question
        session_attributes["current_question"] += 1

        # Always ensure questions object is valid
        questions = session_attributes.get("questions")
        if not questions or "sections" not in questions:
            questions = get_questions()
            if not questions:
                return handler_input.response_builder.speak("Sorry, I couldn’t retrieve the next question.").ask("Could you try again?").response
            session_attributes["questions"] = questions

        # Try next question in same section
        next_question = get_next_question(
            questions, 
            session_attributes["current_section"], 
            session_attributes["current_question"]
        )

        # If no question left in current section, move to next section
        if not next_question:
            session_attributes["current_section"] += 1
            session_attributes["current_question"] = 0
            next_question = get_next_question(questions, session_attributes["current_section"], 0)

        # If no more sections or questions, close the session
        if not next_question:
            closing = questions.get("closing", "Thanks for completing the questionnaire.")
            return handler_input.response_builder.speak(closing).response

        acknowledgements = [
            "Thanks for letting me know. Next: ",
            "Alright, noted.  Next: ",
            "Got it. Next question: ",
            "Understood.  Next question: ",
            "No worries, that's helpful to know. Next: ",
            "Okay, Next: ",
        ]

        # Only insert ack if decision was NO
        if decision == "NO":
            ack = random.choice(acknowledgements)
            return handler_input.response_builder.speak(f"{ack} {next_question['question']}").ask(next_question["question"]).response
        else:
            return handler_input.response_builder.speak(next_question["question"]).ask(next_question["question"]).response


    # Check if the slot name is in the list of free text slots
    elif slot_name in FREE_TEXT_SLOTS:
        # If the slot is free text, we can use Gemini to extract structured information
        final_response = extract_information_with_gemini(current_question_data["question"], final_response)
        # Check if the response is empty after extraction
        if not final_response:
            return handler_input.response_builder.speak("Sorry, I didn't catch that. Could you please repeat?").ask("Could you please repeat?").response

    elif slot_name == "date_of_birth":
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", final_response):
            is_valid, result = validate_with_gemini(slot_name, final_response, current_question_data["question"])
            if not is_valid:
                return handler_input.response_builder.speak(result).ask(result).response
            final_response = result

    elif slot_name == "name":
        parts = final_response.split()
        if len(parts) < 2:
            return handler_input.response_builder.speak("Please tell me your full name.").ask("Could you say your full name?").response
        session_attributes["patient_first_name"] = parts[0].title()
        final_response = final_response.title()

    elif "gender" in slot_name:
        # Gender map
        gender_map = {
            "woman": "female",
            "women": "female",
            "girl": "female",
            "man": "male",
            "boy": "male"
        }
        # Normalize the response to lowercase
        for word, replacement in gender_map.items():
            final_response = final_response.replace(word, replacement)

    elif "email" in slot_name:
        final_response = final_response.replace(" at ", "@").replace(" dot ", ".").strip().lower()
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.[a-z]{2,}$', final_response):
            is_valid, result = validate_with_gemini(slot_name, final_response, current_question_data["question"])
            if not is_valid:
                return handler_input.response_builder.speak(result).ask(result).response


    elif slot_name in ["contact_number", "emergency_contact_phone"]:
        final_response = final_response.replace("oh", "zero")
        digits_only = re.sub(r"\D", "", final_response)
        if digits_only.startswith("0"):
            final_response = "+61" + digits_only[1:]
        if not re.match(r'^\+61[23478]\d{8}$', final_response):
            is_valid, result = validate_with_gemini(slot_name, final_response, current_question_data["question"])
            if not is_valid:
                return handler_input.response_builder.speak(result).ask(result).response

    elif slot_name == "emergency_contact":
        parts = final_response.split()
        if len(parts) < 2:
            return handler_input.response_builder.speak("Could you please provide their full name, including last name?").ask("Could you tell me their full name?").response
        final_response = final_response.title()

    elif slot_name == "emergency_contact_relationship":
        final_response = extract_information_with_gemini(current_question_data["question"], final_response, slot_name)

    elif slot_name in ["surgeries_date", "immunizations_date"]:
        slot_value = re.sub(r"['’](\d{2})", r"20\1", slot_value)
        # First validate the original final_response
        is_valid, result = validate_with_gemini(slot_name, final_response, current_question_data["question"])
        if not is_valid:
            return handler_input.response_builder.speak(result).ask(result).response

        # Now clean the Gemini result
        cleaned = re.sub(r"['’](\d{2})", r"20\1", result)
        cleaned = re.sub(r"\b(\w+)\s+\d{1,2}\s+(20\d{2})", r"\1 \2", cleaned)
        final_response = cleaned.strip()

    else:
        is_valid, result = validate_with_gemini(slot_name, final_response, current_question_data["question"])
        if not is_valid:
            return handler_input.response_builder.speak(result).ask(result).response
        final_response = result

    # If confirmation is needed, store the answer temporarily in the session attributes
    session_attributes["unconfirmed_answer"] = {
        "question_id": current_question_data["question_id"],
        "question_title": current_question_data["question_title"],
        "question_text": current_question_data["question"],
        "response": final_response,
        "section": current_section,
        "question_index": current_question
    }
    # Set flag awaiting_confirmation = True so next time we expect YES/NO
    session_attributes["awaiting_confirmation"] = True

    # Define a list of free text confirmations
    STRUCTURED_CONFIRMATIONS = [
        "Got it. Just to confirm, is your {slot_name} {slot_value}?",
        "Thanks. Just checking: is your {slot_name} {slot_value}?",
        "Understood. Can you confirm that your {slot_name} is {slot_value}?",
        "Thanks. Just to make sure I got it right, is your {slot_name} {slot_value}?",
        "Okay, your {slot_name} is {slot_value}. Is that correct?",
        "Thank you. Did I get your {slot_name} right? Is it {slot_value}?"
    ]
    # Define a list of free text confirmations
    FREE_TEXT_CONFIRMATIONS = [
        "Thanks for sharing. You mentioned: {slot_value}. Is that correct?",
        "You said: {slot_value}. Can I confirm that?",
        "Okay, I heard: {slot_value}. Is that correct?",
        "Just to make sure I got it right, you said: {slot_value}, correct?",
        "You shared: {slot_value}. Did I hear it right?",
        "Got it. You mentioned: {slot_value}. Is that correct?",
        "Thanks. Did I understand correctly: {slot_value}?"
    ]

    # Convert final_response to natural speech if it's a date
    if slot_name in ["surgeries_date", "immunizations_date"]:
        spoken_version = format_date_for_speech(final_response)
    else:
        spoken_version = final_response

    # Ask for Confirmation
    if slot_name in FREE_TEXT_SLOTS:
        confirmation_prompt = random.choice(FREE_TEXT_CONFIRMATIONS).format(slot_value=spoken_version)
    else:
        confirmation_prompt = random.choice(STRUCTURED_CONFIRMATIONS).format(
            slot_name=slot_name.replace('_', ' '), 
            slot_value=spoken_version
        )

    # Add a pause before the confirmation prompt
    confirmation_prompt = add_short_pause(confirmation_prompt, pause_duration_ms=800)
    # Save confirmation prompt for checking later
    session_attributes["confirmation_prompt"] = confirmation_prompt
    # Speak the confirmation prompt and ask for a YES/NO response
    return handler_input.response_builder.speak(confirmation_prompt).ask(confirmation_prompt).response

# Fallback Intent handler
@sb.request_handler(can_handle_func=is_intent_name("AMAZON.FallbackIntent"))
def fallback_intent_handler(handler_input: HandlerInput) -> Response:
    session_attributes = handler_input.attributes_manager.session_attributes

    # If waiting for readiness confirmation, repeat opening message
    if session_attributes.get("waiting_for_ready_confirmation"):
        prompt = session_attributes.get("questions", {}).get("opening", "Are you ready to begin?")
        return handler_input.response_builder.speak("Sorry, I didn't catch that. " + prompt).ask(prompt).response

    # If mid-question, re-prompt the current question
    questions = session_attributes.get("questions", {})
    current_section = session_attributes.get("current_section", 0)
    current_question = session_attributes.get("current_question", 0)
    question_data = get_next_question(questions, current_section, current_question)

    if question_data:
        return handler_input.response_builder.speak("Sorry, I didn't understand that. " + question_data["question"]).ask(question_data["question"]).response

    return handler_input.response_builder.speak("Sorry, I didn't understand. Could you please repeat that?").ask("Could you repeat that?").response

# Lambda handler for deployment
lambda_handler = sb.lambda_handler()
