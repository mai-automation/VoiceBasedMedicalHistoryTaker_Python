from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response, LaunchRequest, IntentRequest, SessionEndedRequest
import pymongo
import os
import logging
from datetime import datetime
import google.generativeai as genai
import json
import random
import re

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
    patient_collection = db['patients']  # Connect to the 'patients' collection in MongoDB

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
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
    
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
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
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
        - Addresses: must include street number, street name, suburb, state, and postcode
        - Names: at least two alphabetic words, capitalised
        - Gender: Accept synonyms of 'male' or 'female' such as woman, man, etc. and convert it to female or male
        - Relationships: Accept common relationships like mother, father, sister, etc.

        Only return one of: VALID, VALID|[Formatted Value], or INVALID|[Reworded Question]
                """.strip()

    # Structured validation rules
    name_rule = "Ensure the response contains at least two alphabetic words representing a typical full name (e.g., 'John Smith')."
    phone_rule = "Convert spoken input to digits. Accept valid Australian numbers with +61 prefix if starting with 0."

    validation_rules = {
        "name": name_rule,
        "emergency_contact": name_rule,
        "date_of_birth": "Convert spoken input into YYYY-MM-DD format. Accept 'eleven november nineteen ninety' as 1990-11-11. Must be a past date.",
        "email": "Ensure it is a valid email. Accept spoken versions like 'john dot doe at gmail dot com'.",
        "gender": "Accept only 'male' or 'female' or common synonyms like 'man', 'woman'.",
        "contact_number": phone_rule,
        "emergency_contact_phone": phone_rule,
        "home_address": "Must include street number, street name, suburb, state, and postcode. Convert spoken numbers/phrases into standard address format.",
        "emergency_contact_relationship": "Accept typical relationships (e.g., mother, father, partner, friend, spouse)."
    }

    # Use specific or fallback rule
    rule = validation_rules.get(slot_name, f"""
    Ensure the patient's response clearly answers the question: '{original_question}'.
    - It should be complete and relevant.
    - If unclear or incomplete, request clarification.
    """)

    prompt = generate_prompt(slot_name, slot_value, original_question, rule)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
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
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
        result = model.generate_content(prompt)
        return result.text.strip() if result.text else user_response  # Default to original if Gemini fails
    except Exception as e:
        logger.error(f"Error extracting structured response with Gemini: {str(e)}")
        return user_response  # Default to original if Gemini fails
    

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
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
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
    

# Handles session end
class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput):
        return isinstance(handler_input.request_envelope.request, SessionEndedRequest)

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


# CaptureAnswerIntent handler
@sb.request_handler(can_handle_func=lambda handler_input:
    isinstance(handler_input.request_envelope.request, IntentRequest) and 
    handler_input.request_envelope.request.intent.name == "CaptureAnswerIntent")
def capture_answer_intent(handler_input: HandlerInput) -> Response:
    logger.info("CaptureAnswerIntent matched successfully.")
    session_attributes = get_session_attributes(handler_input)

    # Handle follow-up answer if we're inside a follow-up chain
    if "current_followup_for" in session_attributes:
        follow_up_key = f"{session_attributes['current_followup_for']}_followups"
        follow_ups = session_attributes.get(follow_up_key, [])

        intent_request = handler_input.request_envelope.request.intent
        slot_value = next((s.value for s in intent_request.slots.values() if s and s.value), None)

        if follow_ups:
            next_follow_up = follow_ups.pop(0)
            session_attributes[follow_up_key] = follow_ups
            logger.info(f"Asking next follow-up: {next_follow_up}")
            return handler_input.response_builder.speak(next_follow_up).ask(next_follow_up).response

        logger.info("All follow-ups completed. Returning to main flow.")
        session_attributes.pop("current_followup_for", None)

    # Initialise session if not yet done
    if not session_attributes.get('questions'):
        questions = get_questions()
        if not questions:
            return handler_input.response_builder.speak(retrieve_error_mg).ask(retrieve_error_mg).response

        session_attributes['questions'] = questions
        session_attributes['current_section'] = 0
        session_attributes['current_question'] = 0
        session_attributes['session_start'] = datetime.utcnow()
        session_attributes['session_id'] = get_next_sequence("session_id")
        session_attributes['patient_id'] = get_next_sequence("patient_id")
        session_attributes['patient_data'] = {}

        next_question = get_next_question(questions, 0, 0)
        speak_output = f"Great! Let me start by asking your personal information. {next_question['question']}"
        return handler_input.response_builder.speak(speak_output).ask(next_question['question']).response

    current_section = session_attributes.get('current_section', 0)
    current_question = session_attributes.get('current_question', 0)
    questions = session_attributes['questions']
    current_question_data = get_next_question(questions, current_section, current_question)
    if not current_question_data:
        return handler_input.response_builder.speak("No further questions. Thank you.").response

    slot_name = current_question_data.get("question_title", "response")
    intent_request = handler_input.request_envelope.request.intent
    slot = intent_request.slots.get(slot_name)
    slot_value = slot.value if slot else None

    if not slot_value:
        for key, alt_slot in intent_request.slots.items():
            if alt_slot and alt_slot.value:
                logger.info(f"[Fallback] Using slot '{key}' with value: {alt_slot.value}")
                slot_value = alt_slot.value
                break

    logger.info(f"[DEBUG] Slot name: {slot_name}")
    logger.info(f"[DEBUG] Slot value: {slot_value}")

    if slot_value:
        try:
            if is_repeat_request(slot_value, current_question_data["question"]):
                logger.info(f"Gemini detected confusion or repeat request from: '{slot_value}'")
                rephrased = get_rephrased_question(current_question_data["question"])
                if rephrased:
                    logger.info(f"Rephrased question: {rephrased}")
                    return handler_input.response_builder.speak(rephrased).ask(rephrased).response
                else:
                    logger.warning("Gemini could not rephrase the question. Falling back to original.")
                    return handler_input.response_builder.speak(
                        f"Let me repeat that. {current_question_data['question']}").ask(current_question_data["question"]).response

        except Exception as e:
            logger.error(f"Error during repeat request detection or rephrasing: {e}")

    if not slot_value:
        return handler_input.response_builder.speak(f"I didn't get that. Could you please repeat your {slot_name.replace('_', ' ')}?").ask(f"Sorry, but can you repeat your {slot_name.replace('_', ' ')}?").response

    if slot_name == "name":
        slot_value = slot_value.title()
        parts = slot_value.strip().split()
        if len(parts) < 2:
            return handler_input.response_builder.speak("Please tell me your full name, including your first and last name.").ask("Could you repeat your full name?").response
        session_attributes["patient_first_name"] = slot_value.split()[0]

    if "email" in slot_name:
        slot_value = slot_value.replace(" at ", "@").replace(" dot ", ".")

    if "home_address" in slot_name:
        slot_value = slot_value.strip(",")

    if "gender" in slot_name:
        slot_value = slot_value.replace("woman", "female").replace("man", "male")

    if slot_name == "emergency_contact_relationship":
        slot_value = slot_value.strip("she's").strip("he's").strip("my").strip("is").strip("are").strip("it's").strip("they're")

    if "medical" in slot_name:
        extracted = extract_information_with_gemini(current_question_data["question"], slot_value)
        logger.info(f"Extracted structured relationship: {extracted}")
        slot_value = extracted

    session_attributes["unconfirmed_answer"] = {
        "question_id": current_question_data["question_id"],
        "question_title": current_question_data["question_title"],
        "question_text": current_question_data["question"],
        "response": slot_value,
        "section": current_section,
        "question_index": current_question
    }
    session_attributes["awaiting_confirmation"] = True

    confirmation_prompt = random.choice([
        "Got it. Just to confirm, is your {slot_name} {slot_value}?",
        "Understood. Can you confirm that your {slot_name} is {slot_value}?",
        "Thanks. Just to confirm, is your {slot_name} {slot_value}?"
    ]).format(slot_name=slot_name.replace("_", " "), slot_value=slot_value)

    logger.info(f"Asking for confirmation: {confirmation_prompt}")

    return handler_input.response_builder.speak(confirmation_prompt).ask(confirmation_prompt).response


# YesIntent handler - Confirmation handler for user responses
@sb.request_handler(can_handle_func=lambda handler_input:
    isinstance(handler_input.request_envelope.request, IntentRequest) and
    handler_input.request_envelope.request.intent.name == "AMAZON.YesIntent")
def yes_intent_handler(handler_input: HandlerInput) -> Response:
    session_attributes = get_session_attributes(handler_input)
    answer = session_attributes.get("unconfirmed_answer")

    # If 'yes' is said before any confirmation is expected — treat it as readiness
    if not answer:
        logger.info("No unconfirmed answer found. Treating this as readiness confirmation.")

        if not session_attributes.get("questions"):
            questions = get_questions()
            if not questions:
                return handler_input.response_builder.speak(retrieve_error_mg).ask(retrieve_error_mg).response

            session_attributes['questions'] = questions
            session_attributes['current_section'] = 0
            session_attributes['current_question'] = 0
            session_attributes['session_start'] = datetime.utcnow()
            session_attributes['session_id'] = get_next_sequence("session_id")
            session_attributes['patient_id'] = get_next_sequence("patient_id")
            session_attributes['patient_data'] = {}

            first_question = get_next_question(questions, 0, 0)
            speak_output = f"Great! Let me start by asking your personal information. {first_question['question']}"
            return handler_input.response_builder.speak(speak_output).ask(first_question['question']).response

        return handler_input.response_builder.speak("Thanks for confirming. Let's continue.").ask("Could you please repeat that?").response

    session_attributes.pop("unconfirmed_answer", None)
    session_attributes["awaiting_confirmation"] = False

    # Continue to confirmation-based flow
    return continue_question_flow(handler_input, answer)


# NoIntent handler - Rejection handler for user responses
@sb.request_handler(can_handle_func=lambda handler_input:
    isinstance(handler_input.request_envelope.request, IntentRequest) and
    handler_input.request_envelope.request.intent.name == "AMAZON.NoIntent")
def no_intent_handler(handler_input: HandlerInput) -> Response:
    session_attributes = get_session_attributes(handler_input)
    answer = session_attributes.get("unconfirmed_answer")

    if not answer:
        logger.info("No unconfirmed answer. Treating 'no' as user not ready.")
        return handler_input.response_builder.speak(
            "That's okay. Just let me know when you're ready to begin."
        ).ask(
            "You can say 'I'm ready' to start when you're ready."
        ).response

    if not isinstance(answer, dict) or "question_text" not in answer:
        logger.warning("Malformed or missing unconfirmed answer.")
        session_attributes["awaiting_confirmation"] = False
        return handler_input.response_builder.speak(
            "Sorry, I lost track of your previous answer. Could you repeat it?"
        ).ask("Can you say that again?").response

    question_text = answer.get("question_text")
    raw_response = answer.get("response")

    # If this is a yes/no question (like 'surgeries', 'allergies'), treat 'No' as the real answer
    yes_no_questions = ["medical_conditions", "surgeries", "medications", "allergies", "family_history", "smoking_status", "alcohol_consumption", "exercise", "immunizations", "previous_medical_records", "children", "family_medical_history", "hereditary_conditions"]

    if answer.get("question_title") in yes_no_questions and raw_response.lower() in ["no", "nope", "nah"]:
        logger.info(f"Interpreting 'No' as actual answer to yes/no medical question.")
    else:
        # Treat as rejection of the proposed answer → re-ask question
        session_attributes["awaiting_confirmation"] = False
        return handler_input.response_builder.speak(
            f"Okay, let's try again. {question_text}"
        ).ask(question_text).response

    # Save 'No' as the final answer
    question_id = answer["question_id"]
    question_title = answer["question_title"]
    current_section = answer["section"]
    current_question = answer["question_index"]
    session_attributes["current_question"] = current_question
    session_attributes["current_section"] = answer["section"]

    final_response = "No"
    patient_data = session_attributes.get("patient_data", {})
    patient_data[question_id] = (question_title, final_response)
    session_attributes["patient_data"] = patient_data
    save_patient_data(session_attributes)

    # Clear follow-up states if they exist
    session_attributes.pop("follow_up_pending", None)
    session_attributes.pop("followups", None)
    session_attributes.pop("followup_index", None)

    # Move to next main question
    session_attributes["current_question"] += 1
    questions = session_attributes["questions"]
    next_question = get_next_question(questions, current_section, session_attributes["current_question"])

    if next_question:
        return handler_input.response_builder.speak(f"Thanks. {next_question['question']}").ask(next_question["question"]).response

    # Try next section if available
    session_attributes["current_section"] += 1
    session_attributes["current_question"] = 0
    next_question = get_next_question(questions, session_attributes["current_section"], 0)

    if next_question:
        return handler_input.response_builder.speak(f"Thanks. Let's continue. {next_question['question']}").ask(next_question["question"]).response

    closing_statement = questions.get("closing", "That's the end of the questions. Thank you!")
    return handler_input.response_builder.speak(closing_statement).response


# Continue the question flow after confirmation
def continue_question_flow(handler_input: HandlerInput, answer: dict) -> Response:
    session_attributes = get_session_attributes(handler_input)

    question_id = answer["question_id"]
    question_title = answer["question_title"]
    question_text = answer["question_text"]
    raw_response = answer["response"]
    current_section = answer["section"]
    current_question = answer["question_index"]
    slot_name = question_title
    questions = session_attributes["questions"]

    logger.info(f"Confirmed response for {question_id} ({slot_name}): {raw_response}")

    final_response = raw_response  # Default unless changed by validation

    # Handle follow-ups for Yes-type answers
    if slot_name in ["medical_conditions", "surgeries", "medications", "allergies", "family_history"] and raw_response.lower() in ["yes", "yeah", "yep"]:
        followups = get_next_question(questions, current_section, current_question).get("follow_up", [])
        if followups:
            session_attributes["follow_up_pending"] = True
            session_attributes["followups"] = followups
            session_attributes["followup_index"] = 0
            followup_q = followups[0]
            logger.info(f"Asking follow-up question: {followup_q}")
            return handler_input.response_builder.speak(followup_q["question"]).ask(followup_q["question"]).response

    # If we are in a follow-up sequence, keep asking follow-ups
    if session_attributes.get("follow_up_pending"):
        followups = session_attributes.get("followups", [])
        index = session_attributes.get("followup_index", 0)
        if index < len(followups):
            followup_q = followups[index]
            question_id = followup_q["question_id"]
            question_title = followup_q["question_title"]
            question_text = followup_q["question"]
            slot_name = question_title
            raw_response = answer["response"]

            patient_data = session_attributes.get("patient_data", {})
            patient_data[question_id] = (slot_name, raw_response)
            session_attributes["patient_data"] = patient_data
            save_patient_data(session_attributes)

            logger.info(f"Saved follow-up: {question_id} → {raw_response}")

            session_attributes["followup_index"] = index + 1
            if session_attributes["followup_index"] < len(followups):
                next_followup = followups[session_attributes["followup_index"]]
                return handler_input.response_builder.speak(next_followup["question"]).ask(next_followup["question"]).response
            else:
                session_attributes.pop("follow_up_pending", None)
                session_attributes.pop("followups", None)
                session_attributes.pop("followup_index", None)

    # Save main response
    patient_data = session_attributes.get("patient_data", {})
    patient_data[question_id] = (slot_name, final_response)
    session_attributes["patient_data"] = patient_data
    save_patient_data(session_attributes)

    logger.info(f"Saved main answer for {slot_name}: {final_response}")

    patient_first_name = session_attributes.get("patient_first_name", "there")

    speak_output = random.choice([
        "Thanks {patient_first_name}! I've saved your {slot_name}.",
        "Got it! Your {slot_name} has been recorded.",
        "Thanks {patient_first_name}! Your {slot_name} has been saved successfully.",
        "Okay, I've noted your {slot_name}.",
        "Great! Your {slot_name} is now stored.",
        "Understood! Your {slot_name} has been saved.",
        "Your {slot_name} has been updated. Thank you {patient_first_name}!"
    ]).format(patient_first_name=patient_first_name, slot_name=slot_name.replace('_', ' '))

    # Move to next question
    session_attributes["current_question"] += 1
    next_question = get_next_question(questions, current_section, session_attributes["current_question"])

    if next_question:
        speak_output += f"<break time='1s'/> {next_question['question']}"
        return handler_input.response_builder.speak(speak_output).ask(next_question["question"]).response

    # If no more questions in this section, go to next section
    session_attributes["current_section"] += 1
    session_attributes["current_question"] = 0
    next_question = get_next_question(questions, session_attributes["current_section"], 0)

    if next_question:
        speak_output += f"<break time='1s'/> Let's move to the next section. {next_question['question']}"
        return handler_input.response_builder.speak(speak_output).ask(next_question["question"]).response

    # End of all questions
    closing_statement = questions.get("closing", "That's the end of the questions. Thank you!")
    return handler_input.response_builder.speak(closing_statement).response


# Lambda handler for deployment
lambda_handler = sb.lambda_handler()
