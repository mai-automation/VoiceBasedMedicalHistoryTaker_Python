from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response, LaunchRequest, IntentRequest
import pymongo
import os
import logging
from datetime import datetime
import google.generativeai as genai
import re
import time
import json

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
    return get_db.client['medical_history_db'] # Return the database

db = get_db()
collection = db['questions']

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
    model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
    
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


# Validate user input using Gemini AI
def validate_with_gemini(slot_name: str, slot_value: str, original_question: str) -> tuple[bool, str]:
    """
    Calls Gemini AI to validate user input for structured (DOB, email) and free-text (response) slots.
    Returns:
      - (True, None) if the response is valid
      - (False, reworded_question) if the response needs clarification
    """

    validation_rules = {
        "name": "Ensure the name contains only alphabetic characters and follows a typical full name format.",
        "date_of_birth": "Convert the response into YYYY-MM-DD format if possible. If invalid, request clarification.",
        "gender": "Ensure the response is either 'male' or 'female'.",
        "phone_number": "Ensure the response is a valid phone number (in Australia). If valid, convert spoken numbers into digit format.",
        "response": "Ensure the response is relevant to the question, contains key information, and is not off-topic."
    }

    validation_prompt = f"""
    The patient was asked: "{original_question}"
    Patient Response: "{slot_value}"
    
    - Does the response correctly answer the question? Answer 'VALID' or 'INVALID'.
    - If the response can be formatted (e.g., date, phone number, email, home address or name correction), return "VALID|[Formatted Value]".
    - If the response is valid but doesn't need formatting, return "VALID".
    - If INVALID, rephrase the original question to make it simpler/clearer to help the patient understand and ensure they provide all necessary details. In this case, the format must be: "INVALID|[Reworded Question]"
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        result = model.generate_content(validation_prompt)
        response_text = result.text.strip()

        logger.info(f"Gemini validation response: {response_text}")

        # Ensure a valid format
        if response_text.startswith("VALID|"):
            formatted_value = response_text.split("|", 1)[1]  # Extract formatted response
            return True, formatted_value

        elif response_text.startswith("INVALID|"):
            reworded_question = response_text.split("|", 1)[1]  # Extract reworded question
            return False, reworded_question  # Ask the user again

        else:
            logger.error(f"Unexpected response format from Gemini: {response_text}")
            return True, slot_value  # Default to original if unexpected

    except Exception as e:
        logger.error(f"Error validating response with Gemini: {str(e)}")
        return True, slot_value  # Default to original response if Gemini fails


def extract_information_with_gemini(question: str, user_response: str) -> str:
    """
    Uses Gemini AI to extract structured details from free-text answers.
    Example: "I have high blood pressure and diabetes" → Extracts "hypertension, diabetes".
    """
    prompt = f"""
    The patient was asked: "{question}"
    The patient responded: "{user_response}"
    
    Extract the key medical details in a **concise** format (e.g., "hypertension, diabetes").
    If no structured details can be extracted, return the original response.
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-pro-exp-02-05")
        result = model.generate_content(prompt)
        return result.text.strip() if result.text else user_response  # Default to original if Gemini fails
    except Exception as e:
        logger.error(f"Error extracting structured response with Gemini: {str(e)}")
        return user_response  # Default to original if Gemini fails


# Initialise SkillBuilder (Manages Alexa skill handlers)
sb = SkillBuilder()

# Error message when data retrieval fails
retrieve_error_mg = "Sorry, I couldn't retrieve the questions. Please try again later."

# Retrieve questions from MongoDB
def get_questions():
    """Fetches all questions from MongoDB"""
    try:
        questions_doc = collection.find_one()  # Fetch the first document in the 'questions' collection
        if questions_doc is None:
            logger.info("No document found in 'questions' collection.")
            return None
        logger.info(f"Retrieved questions: {questions_doc}")
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
    patient_collection = db['patients']  # Connect to the 'patients' collection in MongoDB

    # Extract session_id and patient_id from session attributes
    session_id = session_attributes.get("session_id")
    patient_id = session_attributes.get("patient_id")

    # Define the base query to find the patient's session document
    query = {"session_id": session_id, "patient_id": patient_id}

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
                "session_info.session_end": datetime.utcnow(),  # Set session end time to current time

                # Update the existing response where question_id matches
                "response.$.question": question_text,  # Update the question text if needed
                "response.$.response": answer,  # Update the stored response
                "response.$.time": datetime.utcnow()  # Update the response timestamp
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
                            "time": datetime.utcnow()  # Store the timestamp
                        }
                    }
                },
                upsert=True  # Create a new document if it doesn’t exist
            )

    # Log a success message indicating that data has been saved/updated
    logger.info(f"Patient data saved/updated successfully for session {session_id}.")


# Handles session end
class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput):
        return handler_input.request_envelope.request.object_type == "SessionEndedRequest"

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Session ended.")
        return handler_input.response_builder.response  

sb.add_request_handler(SessionEndedRequestHandler())

# LaunchRequest handler
@sb.request_handler(can_handle_func=lambda handler_input:
                    isinstance(handler_input.request_envelope.request, LaunchRequest))
def launch_request_handler(handler_input: HandlerInput) -> Response:
    """Handles the launch of the Alexa skill and starts asking questions"""
    questions = get_questions()
    session_attributes = get_session_attributes(handler_input)

    if questions:
        session_attributes.update({
            'current_section': 0,
            'current_question': 0,
            'session_start': datetime.utcnow(),
            'session_id': get_next_sequence("session_id"),
            'patient_id': get_next_sequence("patient_id")
        })
        speak_output = questions['opening']
    else:
        speak_output = retrieve_error_mg

    return handler_input.response_builder.speak(speak_output).ask(speak_output).response

# Log full request payload inside the handler
def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to a string format
    raise TypeError(f"Type {type(obj)} not serializable")


# StartMedicalHistoryIntent handler (Handles the "are you ready" confirmation and asks the first question)
@sb.request_handler(can_handle_func=lambda handler_input:
                    isinstance(handler_input.request_envelope.request, IntentRequest) and 
                    handler_input.request_envelope.request.intent.name == "StartMedicalHistoryIntent")
def start_medical_history_intent(handler_input: HandlerInput) -> Response:
    session_attributes = get_session_attributes(handler_input)

    # Get questions from the database
    questions = get_questions()

    # Ensure patient_data is initialised
    if 'patient_data' not in session_attributes:
        session_attributes['patient_data'] = {}

    # Initialise current_section and current_question if not present
    session_attributes['current_section'] = 0
    session_attributes['current_question'] = 0

    # Get the first question from the database
    next_question = get_next_question(questions, session_attributes['current_section'], session_attributes['current_question'])

    if next_question:
        speak_output = f"Great! Let me start by asking your personal information. {next_question['question']}"
    else:
        speak_output = retrieve_error_mg
    
    return handler_input.response_builder.speak(speak_output).ask(speak_output).response


# CaptureAnswerIntent handler
@sb.request_handler(can_handle_func=lambda handler_input:
                    isinstance(handler_input.request_envelope.request, IntentRequest) and 
                    handler_input.request_envelope.request.intent.name == "CaptureAnswerIntent")
def capture_answer_intent(handler_input: HandlerInput) -> Response:
    """Handles user responses, validates using Gemini AI, and moves to the next question."""
    
    logger.info(f"Full request payload: {json.dumps(handler_input.request_envelope.to_dict(), indent=2, default=json_serial)}")
    logger.info("CaptureAnswerIntent matched successfully.")

    session_attributes = get_session_attributes(handler_input)
    current_section = session_attributes.get('current_section', 0)
    current_question = session_attributes.get('current_question', 0)

    # Retrieve questions from MongoDB
    questions = get_questions()
    current_question_data = get_next_question(questions, current_section, current_question)

    if not current_question_data:
        return handler_input.response_builder.speak("No further questions. Thank you.").response

    # Map question to slot name
    slot_mapping = {
        "Full Name": "name",
        "Date of Birth": "date_of_birth",
        "Gender": "gender",
        "Contact Number": "phone_number"
    }

    # Find the appropriate slot name, default to "response"
    slot_name = next((value for key, value in slot_mapping.items() if key in current_question_data["question"]), "response")

    # Extract slot value from the intent request
    intent_request = handler_input.request_envelope.request.intent
    slot = intent_request.slots.get(slot_name)
    slot_value = slot.value if slot else None

    # Format the extracted value based on slot type
    if slot_name == "name" and slot_value:
        patient_name = slot_name
        slot_value = slot_value.title()  # Capitalise name

    if slot_name == "date_of_birth" and slot_value:
        slot_value = slot_value.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")  # Remove suffixes

    # If the question is about an email but is stored in "response", handle formatting
    if "Email" in current_question_data["question"] and slot_value:
        slot_value = slot_value.replace(" at ", "@").replace(" att ", "@").replace(" at mark ", "@").replace(" dot ", ".")

    # If the slot is "response", extract structured data if needed
    if slot_name == "response":
        structured_response = extract_information_with_gemini(current_question_data["question"], slot_value)
        if structured_response:
            slot_value = structured_response  # Store extracted details

    # Ensure a valid response before proceeding
    if not slot_value:
        return handler_input.response_builder.speak(f"I didn't catch that. Could you please provide your {slot_name.replace('_', ' ')} again?").ask(f"Can you repeat your {slot_name.replace('_', ' ')}?").response

    # Get the original question text
    original_question = current_question_data["question"]

    # Call Gemini AI for validation
    is_valid, validated_value_or_reworded_question = validate_with_gemini(slot_name, slot_value, original_question)

    if is_valid:
        # Save validated response
        patient_data = session_attributes.get('patient_data', {})
        patient_data[current_question_data['question_id']] = (  # Store question_id as key
            current_question_data['question_title'],  # question_title
            validated_value_or_reworded_question      # validated response
        )
        session_attributes['patient_data'] = patient_data
        save_patient_data(session_attributes)
        logger.info(f"Patient Data Saved: {json.dumps(patient_data, indent=2, default=str)}")

        speak_output = f"Okay, your {slot_name.replace('_', ' ')} has been saved."

        # Move to next question
        session_attributes['current_question'] += 1
        next_question = get_next_question(questions, current_section, session_attributes['current_question'])

        if next_question:
            speak_output += f"<break time='1s'/> Here is your next question: {next_question['question']}"
        else:
            session_attributes['current_section'] += 1
            session_attributes['current_question'] = 0
            next_question = get_next_question(questions, session_attributes['current_section'], 0)

            if next_question:
                speak_output += f"<break time='1s'/> Thanks {patient_name}. Let's move to the next section. {next_question['question']}"
            else:
                closing_statement = questions.get("closing")
                save_patient_data(session_attributes)  # Save collected data
                return handler_input.response_builder.speak(closing_statement).response

        return handler_input.response_builder.speak(speak_output).ask(speak_output).response  # Ensure response is returned

    else:
        # Response is invalid, retry with reworded question
        speak_output = f"I'm sorry, but {slot_value} doesn't seem valid. {validated_value_or_reworded_question}"
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response  # Ensure response is returned



# Lambda handler for deployment
lambda_handler = sb.lambda_handler()