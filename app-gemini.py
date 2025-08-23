from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
import re
import json
# import google.generativeai as genai
from google import genai
from google.genai import types
import config
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
CORS(app)

# Configuring the path where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini AI API key
# genai.configure(api_key=config.GENAI_API_KEY)

# Global variables
transactions = []
split_transactions = []
uploaded_files = []
text = ""
categories = ""

# Utility: Extract text from PDF
# it will extract the first 2500 characters from the bank statement
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text[:2500]

def call_gemini_ai(prompt):
    try:
        # Initialize the client with the API key from the environment variable
        client = genai.Client(api_key=config.GENAI_API_KEY)
        model = "gemini-1.5-flash"
        
        # Prepare the content with the prompt replacing the placeholder
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ],
            ),
        ]
        
        # Set up the generation configuration
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.85,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )
        
        # Generate content using streaming and concatenate the chunks
        output_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output_text += chunk.text
        
        if not output_text.strip():
            raise ValueError("Empty response from Gemini AI.")
        
        return output_text
    
    except Exception as e:
        raise RuntimeError(f"Error in call_gemini_ai: {e}")

# Utility: Extract format of bank statement
def extract_format():
    global text
    pdf_text = text
    match = re.search(r'.*Date.*', pdf_text, re.MULTILINE)
    return match.group(0) if match else ""

# Utility: Parse transactions
def parse_transactions(text):
    lines = text.split('\n')
    return lines[:500] 

# Route: Clear data
@app.route('/clear_data', methods=['POST'])
def clear_data():
    global transactions, split_transactions, uploaded_files

    transactions = []
    split_transactions = []
    uploaded_files = []

    upload_folder = app.config['UPLOAD_FOLDER']
    for filename in uploaded_files:
        file_path = os.path.join(upload_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({'message': 'Data cleared successfully'}), 200 

def get_split_transactions(transaction_texts):
    transactions_text = "\n".join(transaction_texts)
    
    prompt = f"""
    Based on the following bank transactions, return a Python list of dictionaries as JSON.
    Each dictionary should have:
    - "Date" (STRING): The date of the transaction.
    - "Transaction" (STRING): The transaction description.
    - "Amount" (FLOAT): The transaction amount (negative if debit).
    - "Balance" (FLOAT): The resulting balance after the transaction.

    Return the output as a clean JSON list without any additional formatting or text.

    Transactions:
    {transactions_text}
    """

    result = call_gemini_ai(prompt)

    try:
        # Clean the response by removing code block formatting
        cleaned_result = re.sub(r'```json|```', '', result).strip()

        # Parse the cleaned JSON result
        split_transactions = json.loads(cleaned_result)
        return split_transactions

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from Gemini AI response: {e}")
    
# AI Response Handling in /upload_pdf Route
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global transactions, split_transactions, uploaded_files, text
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part in the request'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No file selected'}), 400

        transactions = []
        split_transactions = []
        uploaded_files = []

        for file in files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                uploaded_files.append(filename)

                # Extract and clean text
                text = extract_text_from_pdf(file_path)
                parsed_transactions = parse_transactions(text)
                transactions.extend(parsed_transactions)

            if transactions:
                try:
                    split_transactions = get_split_transactions(transactions)
                except Exception as e:
                    return jsonify({'error': f"Failed to process transactions: {e}"}), 500

        return jsonify({'transactions': split_transactions, 'text': text}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route: Get uploaded files
@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    global uploaded_files
    try:
        return jsonify(uploaded_files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Route: Get transactions
@app.route('/get_transactions', methods=['GET'])
def get_transactions():
    if isinstance(split_transactions, list):
        return jsonify({'transactions': split_transactions}), 200
    else:
        return jsonify({'error': 'Invalid data format'}), 500

# Function: Return income and expenses    
def get_income_expenses():
    global split_transactions
    income = []
    expenses = []
    for transaction in split_transactions:
        amount_str = transaction['Amount']
        if isinstance(amount_str, str):
            amount_str = amount_str.replace(',', '')
        try:
            amount = float(amount_str)
            if amount < 0:
                expenses.append(amount)
            else:
                income.append(amount)
        except ValueError:
            income = 0
            expenses = 0
            print(f"Warning: Skipping invalid amount value '{amount_str}'")

    return {'income': income, 'expenses': expenses}

# Route: Get total income and expenses
@app.route('/get_income_expenses', methods=['GET'])
def income_expenses():
    try:
        result = get_income_expenses()
        return jsonify(result), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing transactions"}), 500

# Route: Get account balance
@app.route('/get_account_balance', methods=['GET'])
def get_account_balance():
    global split_transactions
    balances = []

    for transaction in split_transactions:
        balance_str = transaction['Balance']

        if isinstance(balance_str, str):
            balance_str = balance_str.replace(',', '')
        
        try:
            balance = float(balance_str)
            balances.append(balance)
        except ValueError:
            balances = 0
            print(f"Warning: Skipping invalid balance value '{balance_str}'")

    return {'balances': balances}

# Route: Get financial summary
@app.route('/get_summary', methods=['GET'])
def get_summary():
    global split_transactions
    summary = get_income_expenses()
    income = sum(summary["income"]) if summary["income"] else 0
    expenses = sum(summary["expenses"]) if summary["expenses"] else 0
    balance_data = get_account_balance()
    balances = balance_data["balances"]
    current_balance = balances[-1] if balances else 0
    initial_balance = balances[0] if balances else 0
    savings = current_balance - initial_balance

    return jsonify({
        'balance': current_balance,
        'income': income,
        'expenses': expenses,
        'savings': savings
    }), 200

# Model 1: Sustainable transactions
import json
import re
from flask import jsonify

@app.route('/sustainable_transactions', methods=['GET'])
def sustainable_transactions():
    global split_transactions

    # Prepare the transactions text as before
    transactions_text = "\n".join(
        [f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions]
    )
    prompt = f"""
    Based on the following bank transactions, provide an estimated sustainability score and a brief explanation as a JSON object.
    Provide the score and the explanation to the best of the understanding of the text; if no significant figures are present,
    provide an average or an estimated value.
    The JSON object must strictly follow this format and contain no additional text or markdown formatting:

    {{"score": <integer between 1 and 100>, "reasoning": "<brief explanation in about 75 words>"}}

    For example, if the transactions show a balanced pattern with minimal waste, the output should be similar to:
    {{"score": 85, "reasoning": "The transactions demonstrate a balanced financial behavior with expenditures in organic stores like XYZ with minimal unnecessary spending, indicating a sustainable pattern. The user effectively manages expenditures and maintains a steady balance over time."}}

    Alternatively, if the transactions indicate unsustainable behavior with excessive or wasteful spending, the output might be:
    {{"score": 35, "reasoning": "The transactions indicate high levels of unsustainable spending like in stores ABC, with frequent impulsive purchases and little evidence of budget management. The user seems to incur excessive costs without maintaining a stable balance, suggesting an unsustainable financial pattern."}}

    Return only the JSON object without any extra commentary.

    Transactions:
    {transactions_text}
    """
    try:
        # Call the AI with the refined prompt
        result = call_gemini_ai(prompt)
        print(f"Raw AI Response: {result}")

        # Enhanced post-processing:
        # Remove any markdown/code block formatting that might be present
        cleaned_result = re.sub(r'```(json)?|```', '', result).strip()

        # Optionally, further cleanup any extraneous characters if needed
        # For example, remove any leading/trailing text that isn't part of the JSON
        json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the AI response.")

        json_text = json_match.group(0)

        # Parse the JSON result
        parsed_result = json.loads(json_text)

        # (Optional) Validate against a simple schema manually
        if not isinstance(parsed_result, dict) or \
           "score" not in parsed_result or \
           "reasoning" not in parsed_result:
            raise ValueError("JSON does not contain the required keys.")


        # Extract the values
        score = parsed_result.get("score", "Unknown")
        reasoning = parsed_result.get("reasoning", "No explanation provided.")

        return jsonify({"score": score, "reasoning": reasoning, 'prompt': prompt})

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return jsonify({"error": "Failed to parse JSON response from Gemini AI"}), 500
    except ValueError as e:
        return jsonify({"error": f"AI response issue: {e}"}), 500


# Model 2: Risk analysis score
@app.route('/risk_analysis', methods=['GET'])
def get_risk_analysis_score():
    global split_transactions

    transactions_text = "\n".join(
        [f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions]
    )

    prompt = f"""
    Analyze the financial risk associated with the following transactions and return the result as a JSON object.
    
    Consider:
    - Companies or individuals involved in transactions
    - Savings, income, and expenses over the period
    
    Return the result in this format:
    {{
        "risk_score": (integer between 1 and 100),
        "reasoning": (brief explanation string)
    }}

    Transactions:
    {transactions_text}
    """

    try:
        result = call_gemini_ai(prompt)
        print(f"Raw AI Response: {result}")

        match = re.search(r'\{.*\}', result, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the AI response.")

        parsed_result = json.loads(match.group(0))
        risk_score = parsed_result.get("risk_score", "Unknown")
        reasoning = parsed_result.get("reasoning", "No reasoning provided.")

        return jsonify({'risk_score': risk_score, 'reasoning': reasoning, 'prompt': prompt})

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return jsonify({'error': 'Failed to parse JSON response from Gemini AI'}), 500
    except ValueError as e:
        print(f"AI response issue: {e}")
        return jsonify({'error': f'AI response issue: {e}'}), 500
    
# Model 3: Calculate and display percentages (Budget Tracking)
@app.route('/calculate_percentages', methods=['GET'])
def calculate_and_display_percentages():
    global split_transactions, text
    if not split_transactions:
        return jsonify({'percentages': [{"category": "No Data", "percentage": 100.0}]})

    transactions_text = "\n".join([f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions])
    response = categorize_transactions(transactions_text)
    
    return jsonify({'percentages': response})


def categorize_transactions(transactions_text):
    global categories

    prompt = f"""
    Given the bank statement transactions below, categorize the expenses into 'Food', 'Entertainment', 
    'Transportation', 'Utilities', etc., and return their percentage of the total expenses.
    
    Transactions:
    {transactions_text}

    Return the response in the following JSON format:
    [
        {{
            "category": "string",
            "percentage": float
        }},
        ...
    ]
    """

    try:
        result = call_gemini_ai(prompt)

        # Extract the JSON from the result
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            categories = json.loads(match.group(0).strip())
            return categories
        else:
            raise ValueError("No valid JSON found in the response")

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Failed to categorize transactions. Error: {e}")
        # Default sample data in case of failure
        sample = [
            {"category": "Food", "percentage": 26.43},
            {"category": "Payments to Individuals", "percentage": 30.12},
            {"category": "Education", "percentage": 46.8},
            {"category": "Miscellaneous", "percentage": 0.02}
        ]
        return sample
    
# Model 4: Chatbot interaction
# Initial system prompt for the ITR chatbot
itr_initial_messages = [
    {
        "role": "system",
        "content": """You are a Smart ITR Filing Chatbot. Your job is to help users determine the most suitable Income Tax Return (ITR) form based on their income heads. 
        Ask questions about their income sources, including salary, business income, rental income, capital gains, and interest income. 
        Provide tax-saving investment suggestions under sections like 80C, 80D, and 80E and give user-friendly responses."""
    }
]

# Initial messages for the financial advisory chatbot
financial_advisory_initial_messages = [
    {
        "role": "system",
        "content": """You are a Financial Advisory System that analyzes the user's bank transactions 
        and provides personalized recommendations. 

        Here is what you should do:
        1) Ask clarifying questions about the user’s financial goals (e.g., saving for a house, investing in stocks).
        2) Based on the user’s responses (like monthly income, timeline, desired purchase), ask for further details.
        3) Suggest a realistic monthly savings plan to achieve the user’s goal within their target timeframe. 
        4) Identify high-expense categories from the transaction data and recommend ways to reduce expenses.
        5) If the user wants to expedite their goal, show how cutting certain expenses or increasing savings 
           can shorten the timeline.
        All the transactions would be in INR (Indian Ruppes) only.
        After your final *human-readable* response, always include a concise JSON section 
        that describes the key data points needed for charting. 
        Make sure the JSON is *clearly marked* and parseable.
        For example:

        (Human-readable summary...)

        [JSON DATA]
        {
            "goal": "string",
            "timelineYears": 10,
            "monthlySavingsTarget": 2000,
            "expenseCuts": [
                {{"category": "Food", "reduction": 0.2}},
                ...
            ]
            'additionalIncomeNeeded': True/ False, 
            'incomeNeededAmount': 1000
        }
        [/JSON DATA]
        Sample JSON Data: {'goal': 'Car Purchase', 'timelineYears': 2, 'monthlySavingsTarget': 20833.33, 'expenseCuts': [{'category': 'Food Delivery', 'reduction': 0.75}, {'category': 'Ride-Hailing', 'reduction': 0.5}], 'additionalIncomeNeeded': True, 'incomeNeededAmount': 1000}
        The JSON block should be a valid JSON object enclosed between [JSON DATA] and [/JSON DATA].
        """
    }
]


# Initial messages for the BOB query chatbot
bob_initial_messages = [
    {
        "role": "system",
        "content": """You are an AI assistant that helps people find information about Bank of Baroda services.
        Focus only on Bank of Baroda information. Do not provide information about other banks.
        If asked about other banks or personal transactions, politely refuse and redirect to Bank of Baroda services."""
    },
    {
        "role": "assistant",
        "content": """For Bank of Baroda related queries, I can provide information based on their official website: https://www.bankofbaroda.in/
        If you don't know the answer, fabricate your own answers and redirect to website at the end.
        If asked about any other bank except Bank of Baroda(BOB), respond with: 'I'm sorry, but I can only provide information about Bank of Baroda.
        How can I assist you with Bank of Baroda services today?
        Always try to promote Bank of Baroda services when appropriate.
        Give it as a detailed list with formatting"""
    }
]

def interact_with_gemini(messages):
    try:
        # Convert the conversation messages into a single prompt string
        conversation_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Initialize the client with the API key from your configuration
        client = genai.Client(api_key=config.GENAI_API_KEY)
        model = "gemini-1.5-flash"
        
        # Prepare the content with the prompt replacing the placeholder
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=conversation_prompt)
                ],
            ),
        ]
        
        # Set up the generation configuration
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.85,
            top_k=40,
            max_output_tokens=1000,
            response_mime_type="text/plain",
        )
        
        # Generate content using streaming and concatenate the chunks
        output_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output_text += chunk.text
        
        if not output_text.strip():
            raise ValueError("Empty response from Gemini AI.")
        
        return output_text

    except Exception as e:
        raise RuntimeError(f"Error in interact_with_gemini: {e}")


# ITR Chatbot
@app.route('/itr_chatbot', methods=['POST'])
def itr_chatbot():
    global itr_initial_messages
    user_query = request.json.get("query", "")
    language = request.json.get("language", "English") 
    print("Language: ", language)
    messages = itr_initial_messages.copy()
    messages.append({
        "role": "system",
        "content": f"Respond in language: {language}"
    })
    messages.append({"role": "user", "content": user_query})
    assistant_message = interact_with_gemini(messages)

    return jsonify({
        "response": assistant_message,
        "messages": messages + [{"role": "assistant", "content": assistant_message}]
    })

# Financial Advisory Chatbot
@app.route('/transaction_chatbot', methods=['POST'])
def transaction_chatbot():
    global transactions, text, financial_advisory_initial_messages, split_transactions

    user_query = request.json.get("query", "")
    language = request.json.get("language", "English")
    print("Language: ", language)

    extracted_text = request.json.get("extractedText", {})
    split_transaction = extracted_text.get("split_transaction", "")
    current_text = split_transaction if split_transaction else text

    messages = financial_advisory_initial_messages.copy()

    messages.append({
        "role": "assistant",
        "content": (
            f"I have the following transaction data:\n{current_text}\n"
            "Let me know your financial goals so I can advise you."
        )
    })

    messages.append({
        "role": "system",
        "content": f"Respond in language: {language}"
    })

    # User query
    messages.append({"role": "user", "content": user_query})

    # Get the model's complete response
    assistant_message = interact_with_gemini(messages)
    print("Assistant Message: ", assistant_message)

    human_readable_text = assistant_message
    chart_data = {}

    # Search for a JSON section enclosed between [JSON DATA] and [/JSON DATA]
    match = re.search(r'\[JSON DATA\](.*?)\[/JSON DATA\]', assistant_message, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        
        # Remove the entire [JSON DATA] ... [/JSON DATA] block from the human-readable text
        human_readable_text = (
            assistant_message[:match.start()] + assistant_message[match.end():]
        ).strip()

        # Attempt to parse the JSON content
        try:
            chart_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, you can log an error or provide fallback data
            print("Warning: Could not parse the chatbot's JSON data.")
            chart_data = {}

    print("Human Readable Text: ", human_readable_text)
    print("Chart Data: ", chart_data)
    print("Messages: ", messages)

    # Return the separated response: human text + structured JSON data
    return jsonify({
        "response": human_readable_text,
        "chartData": chart_data,
        "messages": messages + [{"role": "assistant", "content": assistant_message}]
    })


def scrape_bob_website():
    url = "https://www.bobcard.co.in/credit-card-offers"  # Or a more specific page URL if you know it
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        soup = BeautifulSoup(response.content, "html.parser")

        # Get all the text content of the page (Less structured)
        all_text = soup.get_text(strip=True)  # strip=True removes extra whitespace

        return all_text  # Return the entire text content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# BOB Chatbot
@app.route('/bob_chatbot', methods=['POST'])
def bob_chatbot():
    global bob_initial_messages
    user_query = request.json.get("query", "")
    language = request.json.get("language", "English")
    print("Language: ", language)

    # Scrape the BOB website for fresh info to use in the chatbot's response
    website_info = scrape_bob_website()
    print("Website Info: ", website_info)
    messages = bob_initial_messages.copy()
    messages.append({
        "role": "system",
        "content": f"Respond in language: {language}. Use the following data from BOB website: {website_info}"
    })
    messages.append({"role": "user", "content": user_query})

    # Assuming interact_with_gemini is a function that sends these messages to your AI model
    assistant_message = interact_with_gemini(messages)

    return jsonify({
        "response": assistant_message,
        "messages": messages + [{"role": "assistant", "content": assistant_message}]
    })


# Model 5: Wellness Score
@app.route('/wellness_score', methods=['GET'])
def get_wellness_score():
    global split_transactions

    transactions_text = "\n".join(
        [f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions]
    )

    prompt = f"""
    Analyze the wellness implications of the following transactions based on spending categories and return the result as a JSON object.
    
    Consider:
    - Categories of the transactions such as groceries, fast food, gym memberships which may impact health positively or negatively.
    - Overall spending habits and frequency on unhealthy options like fast food versus healthy options like gym memberships or organic groceries.
    
    Return the result in this format:
    {{
        "wellness_score": (integer between 1 and 100),
        "reasoning": (brief explanation string)
    }}

    Transactions:
    {transactions_text}
    """

    try:
        result = call_gemini_ai(prompt)
        print(f"Raw AI Response: {result}")

        match = re.search(r'\{.*\}', result, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the AI response.")

        parsed_result = json.loads(match.group(0))
        wellness_score = parsed_result.get("wellness_score", "Unknown")
        reasoning = parsed_result.get("reasoning", "No reasoning provided.")

        return jsonify({'wellness_score': wellness_score, 'reasoning': reasoning})

    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return jsonify({'error': 'Failed to parse JSON response from Gemini AI'}), 500
    except ValueError as e:
        print(f"AI response issue: {e}")
        return jsonify({'error': f'AI response issue: {e}'}), 500


# Model 6
# Route: Stock recommendations
@app.route('/recommended_stocks', methods=['GET'])
def recommended_stocks():
    global split_transactions
    if not split_transactions:
        return jsonify({"error": "No transaction data available"}), 400

    transactions_text = "\n".join([f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions])

    prompt = f"""
    Based on the user's spending habits in the transactions below, recommend 5 stocks in the form of a JSON format.
    Each stock recommendation should include:
    - Name(Name)(string)
    - Symbol(Symbol)(String)
    - INR Price(INR Price)(string) (Enter random Price if not)
    - Reason for recommendation(Reason)(string)

    Transactions:
    {transactions_text}

    Give a json.
    """

    try:
        result = call_gemini_ai(prompt)
        if not result:
            return jsonify({"error": "Empty response from AI"}), 500

        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if not match:
            return jsonify({"error": "Failed to extract stock recommendations, no match found"}), 500

        recommendations = json.loads(match.group(0).strip())
        return jsonify({'recommendations': recommendations})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Failed to parse JSON response: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unhandled error: {e}"}), 500

# Model 7
# Route: Retirement Planning
@app.route('/retirement_planning', methods=['POST'])
def retirement_planning():
    global split_transactions  # Using global transaction data
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    if not split_transactions:
        return jsonify({"error": "No transaction data available"}), 400

    transactions_text = "\n".join(
        [f"{t['Date']}, {t['Transaction']}, {t['Amount']}, {t['Balance']}" for t in split_transactions]
    )

    prompt = f"""
    Predict my estimated savings at different years based on the details below:
    
    - Current Age: {data.get('currentAge', 'N/A')}
    - Retirement Age: {data.get('retirementAge', 'N/A')}
    - Marital Status: {data.get('maritalStatus', 'N/A')}
    - Spouse Age: {data.get('spouseAge', 'N/A')}
    - Work Income (Monthly): {data.get('workIncome', 'N/A')}
    - Current Savings: {data.get('currentSaving', 'N/A')}
    - Recent Transactions:
    {transactions_text}

    Provide the output in JSON format as a list of objects where each object has:
    - 'date': Year (starting from 2024, increasing in 2-year intervals until 2066) (integer)
    - 'savings': Cumulative estimated savings for that year. (integer)

    Output must be a JSON array.
    ONLY return a valid JSON array. Do not include any text before or after it.
    """

    try:
        result = call_gemini_ai(prompt)
        if not result:
            return jsonify({"error": "Empty response from AI"}), 500

        # Extract JSON response using regex
        print("AI response:\n", result)
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            parsed_result = json.loads(match.group(0))
            return jsonify({"retirementTracking": parsed_result})
        else:
            return jsonify({"error": "Failed to extract JSON from AI response"}), 500

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode JSON from AI response"}), 500
    except Exception as e:
        return jsonify({"error": f"Unhandled error: {e}"}), 500

if __name__ == '__main__':
    app.run()
