from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalModelEmailClassifier:
    def __init__(self, model_path):
        """
        Initialize the EmailClassifier with the local model path.
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Load the local model and tokenizer.
        """
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def classify_email(self, email_content):
        """
        Classify an email using the local model.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer is not loaded. Call load_model() first.")

        # Prepare the input prompt
        prompt = (
            f"Classify the following email as phishing or ham. "
            f"Provide a score between 0 (very unlikely phishing) to 1 (very likely phishing):\n\n"
            f"{email_content}\n\n"
            f"Response:"
        )

        # Tokenize the input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate the model's response
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract numerical score from the response
        try:
            score = float(response.split()[-1])
            score = min(max(score, 0), 1)  # Ensure the score is within 0-1
        except ValueError:
            score = 0.5  # Default to neutral if parsing fails

        return score, response

    @staticmethod
    def main():
        """
        Main method to demonstrate email classification.
        """
        # Path to the local model
        MODEL_PATH = "/Users/houda/Downloads/lora_finetuned_model/"  # Replace with the actual path to your model

        # Initialize the classifier
        classifier = LocalModelEmailClassifier(MODEL_PATH)
        classifier.load_model()

        # Example email content
        email_content = """
        Dear User,
        Your account has been compromised. Click here to reset your password immediately: http://fakeurl.com
        """

        # Classify the email
        print("Classifying email...")
        score, response = classifier.classify_email(email_content)
        print(f"Phishing Likelihood Score: {score:.2f}")
        print(f"Model Response: {response}")


# Run the main method
if __name__ == "__main__":
    LocalModelEmailClassifier.main()