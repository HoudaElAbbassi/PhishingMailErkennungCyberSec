import re

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

class EmailClassifier:
    def __init__(self, api_key, endpoint, deployment_name, prompt, email):
        """
        Initialize the EmailClassifier with Azure OpenAI credentials.
        """
        self.prompt = prompt
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.llm = None
        self.chain = None
        self.email = email

    def initialize_model(self):
        """
        Initialize the Azure OpenAI LLM with LangChain.
        """
        print("Initializing Azure OpenAI LLM...")
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            azure_deployment=self.deployment_name,
            api_version="2023-03-15-preview",
        )

        # Define the prompt template for email classification
        prompt_template = PromptTemplate(
            input_variables=["email_content"],
            template=(
                self.prompt
            ),
        )

        # Create the LLM chain
        self.chain = prompt_template | self.llm
        # self.chain.invoke("Test")  # Warm-up the chain

    def classify_email(self, email_content):
        """
        Classify an email and return the likelihood score (0-1).
        """
        if not self.chain:
            raise RuntimeError("Model chain is not initialized. Call initialize_model() first.")

        # Run the chain
        response = self.chain.invoke({"email_content": email_content})

        # Extract the content from the AIMessage object
        if isinstance(response, AIMessage):
            response_content = response.content  # Get the textual response
        else:
            raise ValueError("Unexpected response type from LLM chain.")

        # Parse the response for the phishing score
        try:
            # Use regex to search for "Score: <number>" in the response
            match = re.search(r"Score:\s*([\d.]+)", response_content)
            if match:
                # Convert the captured number to a float
                score = float(match.group(1))
                return score,response_content
            else:
                raise ValueError("Score not found in response.")
        except (ValueError, IndexError) as e:
            print(f"Error extracting score: {e}")
            # Default score if parsing fails
            return 0.5

        return score, response_content

    @staticmethod
    def main(api_key, endpoint, deployment_name, prompt, email):
        """
        Main method to demonstrate email classification.
        """


        # Initialize the classifier
        classifier = EmailClassifier(api_key, endpoint, deployment_name, prompt, email)
        classifier.initialize_model()

        # Example email content
        email_content = """
        Dear User,
        how are you? Link Pleaaase Klick
        """

        # Classify the email
        print("Classifying email...")
        score, response = classifier.classify_email(email)
        print(f"Phishing Likelihood Score: {score}")
        print(f"Model Response: {response}")
        return response, score


# Run the main method
if __name__ == "__main__":
    EmailClassifier.main()
