import json


from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate


class EmailClassifier:
    def __init__(self, api_key, endpoint, deployment_name, email,llm,prompt_template):
        """
        Initialize the EmailClassifier with Azure OpenAI credentials.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.email = email
        self.llm= llm
        self.chain =  prompt_template | llm

    @staticmethod
    def initialize_model(self,llm):
        """
        Initialize the Azure OpenAI LLM with LangChain.
        """
        print("Initializing Azure OpenAI LLM...")

       # Define the prompt template for email classification
        prompt_template = PromptTemplate(
            input_variables=["email_content"],
            template="""
Below are examples of email classification with step-by-step reasoning:
Example 1:
Email: "Your account has been locked. Click here to unlock."
Step-by-step analysis:
1. The sender's email address is suspicious and does not match the official domain.
2. The language is urgent and manipulative ("Your account has been locked").
3. The link points to a non-official domain ("Click here to unlock").
4. URL uses unusual subdomains or parameters.
5. URL domain does not match common patterns for the claimed organization.
Result: Phishing, Score: 0.9

Example 2:
Email: "Here is your invoice for the month."
Step-by-step analysis:
1. The sender's email address matches the official company domain.
2. The language is neutral and does not create urgency.
3. No suspicious links are present.
4. URL domain matches official company patterns.
5. URL contains expected secure parameters.
Result: Not Phishing, Score: 0.1

You are an AI tasked with classifying emails for phishing detection. Analyze the email below and provide a JSON response with the following keys:
- subject_score
- sender_score
- content_score
- url_score
- subject_analysis
- sender_analysis
- content_analysis
- url_analysis

Response format (strictly JSON):
{{
    "subject_score": 0.0,
    "sender_score": 0.0,
    "content_score": 0.0,
    "url_score": 0.0,
    "subject_analysis": "string",
    "sender_analysis": "string",
    "content_analysis": "string",
    "url_analysis": "string"
}}

Classify the following email:
Email: {email_content}
Step-by-step analysis:
1. Analyze the sender's email address.
2. Check for misleading or urgent language.
3. Evaluate any URLs in the email for unusual patterns or parameters.
4. Compare the URL domain with common patterns for trusted organizations.
5. Assign a score based on the findings and classify the email as Phishing or Not Phishing.
Provide the final classification and phishing likelihood score along with scores for each component (subject, sender, content, url).
            """,
        )

        # Create the LLM chain
        self.chain = prompt_template | llm

    def classify_email(self,llm):
        """
        Classify an email and return detailed scores and analysis dynamically.
        """
        if not self.chain:
            raise RuntimeError(
                "Model chain is not initialized. Call initialize_model() first.")

        # Run the chain
        response = self.chain.invoke({"email_content": self.email})
        # Debugging: Zeige die rohe Antwort an
        print("Raw LLM Response:", response)
        # Extract the content from the AIMessage object
        if isinstance(response, AIMessage):
            response_content = response.content  # Get the textual response
        else:
            raise ValueError("Unexpected response type from LLM chain.")

        # Debugging: Zeige den extrahierten Inhalt an
        print("Extracted Response Content:", response_content)

        # Parse the response dynamically for detailed scores and analysis
        try:
            # Ensure response_content is a valid JSON string
            # Remove any leading/trailing spaces
            response_content = response_content.strip()
            response_data = json.loads(response_content)
            return response_data
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw LLM response for debugging: {response_content}")
            raise

    @staticmethod
    def main(api_key, endpoint, deployment_name, email,llm):
        """
        Main method to demonstrate email classification.
        """
        prompt=  prompt_template = PromptTemplate(
            input_variables=["email_content"],
            template="""
Below are examples of email classification with step-by-step reasoning:
Example 1:
Email: "Your account has been locked. Click here to unlock."
Step-by-step analysis:
1. The sender's email address is suspicious and does not match the official domain.
2. The language is urgent and manipulative ("Your account has been locked").
3. The link points to a non-official domain ("Click here to unlock").
4. URL uses unusual subdomains or parameters.
5. URL domain does not match common patterns for the claimed organization.
Result: Phishing, Score: 0.9

Example 2:
Email: "Here is your invoice for the month."
Step-by-step analysis:
1. The sender's email address matches the official company domain.
2. The language is neutral and does not create urgency.
3. No suspicious links are present.
4. URL domain matches official company patterns.
5. URL contains expected secure parameters.
Result: Not Phishing, Score: 0.1

You are an AI tasked with classifying emails for phishing detection. Analyze the email below and provide a JSON response with the following keys:
- subject_score
- sender_score
- content_score
- url_score
- subject_analysis
- sender_analysis
- content_analysis
- url_analysis

Response format (strictly JSON):
{{
    "subject_score": 0.0,
    "sender_score": 0.0,
    "content_score": 0.0,
    "url_score": 0.0,
    "subject_analysis": "string",
    "sender_analysis": "string",
    "content_analysis": "string",
    "url_analysis": "string"
}}

Classify the following email:
Email: {email_content}
Step-by-step analysis:
1. Analyze the sender's email address.
2. Check for misleading or urgent language.
3. Evaluate any URLs in the email for unusual patterns or parameters.
4. Compare the URL domain with common patterns for trusted organizations.
5. Assign a score based on the findings and classify the email as Phishing or Not Phishing.
Provide the final classification and phishing likelihood score along with scores for each component (subject, sender, content, url).
            """,
        )

        # Initialize the classifier
        classifier = EmailClassifier(api_key, endpoint, deployment_name, email,llm,prompt_template=prompt)#,llm)
        #classifier.initialize_model(llm)

        # Classify the email
        print("Classifying email...")
        response = classifier.classify_email(llm)
        print(f"Model Response: {response}")
        return response, response

    # @staticmethod
    # def main(api_key, endpoint, deployment_name, email):
    #     """
    #     Main method to demonstrate email classification.
    #     """
    #     # Initialize the classifier
    #     classifier = EmailClassifier(api_key, endpoint, deployment_name, email)
    #     classifier.initialize_model()

    #     # Classify the email
    #     print("Classifying email...")
    #     score, response = classifier.classify_email()
    #     print(f"Phishing Likelihood Score: {score}")
    #     print(f"Model Response: {response}")
    #     return response, score


# # Example usage
# if _name_ == "_main_":
#     API_KEY = "your-api-key"
#     ENDPOINT = "your-endpoint"
#     DEPLOYMENT_NAME = "your-deployment-name"

#     EMAIL_CONTENT = """
#     Dear User,
#     Please click on the link below to verify your account:
#     https://phishing-link.example.com
#     """

#     EmailClassifier.main(API_KEY, ENDPOINT, DEPLOYMENT_NAME, EMAIL_CONTENT)