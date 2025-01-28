import re
from email import message_from_bytes
from typing import Dict

import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, File, UploadFile
from huggingface_hub._webhooks_server import JSONResponse
from requests.compat import chardet

from checkWithLLM import EmailClassifier
from checkWithLoRA import EmailClassifierLora

app = FastAPI()

@app.post("/checkMailLoRa")
async def upload_lora( email:str, betreff:str, url:str, absender:str):
    # Instantiate the class with the provided parameters
    emailClassify = EmailClassifierLora(email, betreff, url, absender)
    # Call the classify method
    result = emailClassify.classify(emailClassify,email, betreff, url, absender)
    return result

def clean_email_content(content: str) -> str:
    """
    Cleans email content by removing unnecessary line breaks, spaces, and artifacts.
    """
    # Remove metadata headers (From, Sent, To, Subject) if already parsed
    content = re.sub(r"(From:.|Sent:.|To:.|Subject:.)", "", content, flags=re.IGNORECASE)

    # Remove line break artifacts
    content = re.sub(r"(\r\n|\n|\r)", " ", content)

    # Fix broken words (e.g., "th\ne" -> "the")
    content = re.sub(r"(?<=[a-zA-Z])\s+(?=[a-zA-Z])", "", content)

    # Remove extra whitespace
    content = re.sub(r"\s{2,}", " ", content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content

def parse_eml(file: bytes) -> Dict:
    """Parse .eml file to extract relevant details."""
    message = message_from_bytes(file)
    subject = message.get("Subject", "No Subject")
    sender = message.get("From", "No Sender")
    urls = []
    body = ""

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
            elif content_type == "text/html":
                html_content = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html_content, "html.parser")
                body = soup.get_text(separator="\n").strip()
                urls = [a["href"] for a in soup.find_all("a", href=True)]
    else:
        body = message.get_payload(decode=True).decode("utf-8", errors="ignore")

    return {"betreff": subject, "absender": sender, "url": urls, "email": clean_email_content(body)}

def parse_htm(file: bytes) -> Dict:
    """Parse .htm file to extract relevant details."""
    soup = BeautifulSoup(file, "html.parser")

    # Extract "Subject"
    subject_tag = soup.find(text=re.compile(r"Subject:", re.IGNORECASE))
    subject = subject_tag.split(":")[1].strip() if subject_tag else "No Subject"

    # Extract "From"
    from_tag = soup.find(text=re.compile(r"From:", re.IGNORECASE))
    sender = from_tag.split(":")[1].strip() if from_tag else "No Sender"

    # Extract "To"
    to_tag = soup.find(text=re.compile(r"To:", re.IGNORECASE))
    recipient = to_tag.split(":")[1].strip() if to_tag else "No Recipient"

    # Extract URLs
    urls = [a["href"] for a in soup.find_all("a", href=True)]

    # Extract Email Body
    body = soup.get_text(separator="\n").strip()

    return {
        "betreff": subject,
        "absender": sender,
        "empfänger": recipient,
        "url": urls,
        "email": clean_email_content(body),
    }

def parse_txt(file: bytes) -> Dict:
    """Parse .txt file to extract details."""
    encoding = chardet.detect(file)["encoding"]  # Detect encoding
    text_content = file.decode(encoding, errors="ignore")

    # Extract "Subject"
    subject_match = re.search(r"Subject:\s*(.+)", text_content, re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else "No Subject"

    # Extract "From"
    from_match = re.search(r"From:\s*(.+)", text_content, re.IGNORECASE)
    sender = from_match.group(1).strip() if from_match else "No Sender"

    # Extract "To"
    to_match = re.search(r"To:\s*(.+)", text_content, re.IGNORECASE)
    recipient = to_match.group(1).strip() if to_match else "No Recipient"

    # Extract URLs
    urls = re.findall(r"http[s]?://\S+", text_content)

    return {
        "betreff": subject,
        "absender": sender,
        "empfänger": recipient,
        "url": urls,
        "email": clean_email_content(text_content),
    }

@app.post("/uploadMail")
async def upload_mail(file: UploadFile):
    """
    Endpoint to upload a mail file (eml, pdf, text, html) and extract the subject (betreff),
    sender (absender), URLs, and email content.
    """
    content =  file.file.read()
    # Handle different file types based on file extension
    if file.filename.endswith(".eml"):
        result = parse_eml(content)
    #elif file.filename.endswith(".pdf"):
    #    result = parse_pdf(content)
    elif file.filename.endswith(".txt"):
        result = parse_txt(content)
    elif file.filename.endswith(".html") or file.filename.endswith(".htm"):
        result = parse_htm(content)
    else:
        return JSONResponse(
            content={"error": "Unsupported file type. Please upload .eml, .pdf, .txt, or .html files."},
            status_code=400,
        )
    return result

@app.get("/checkMailLLM")
async def check_mail_llm(
    api_key: str = Query(..., description="Your Azure OpenAI API key"),
    endpoint: str = Query(..., description="Your Azure OpenAI endpoint"),
    deployment_name: str = Query(...,
                                 description="Your Azure OpenAI deployment name"),
    subject: str = Query(..., description="The subject of the email"),
    sender: str = Query(..., description="The sender of the email"),
    recipient: str = Query(..., description="The recipient of the email"),
    content: str = Query(..., description="The content of the email"),
    url: str = Query(..., description="The URL included in the email")
):
    """
    Check if an email is phishing using an LLM.
    """
    # Combine the fields into an email content string
    def sanitize_content(content: str) -> str:
        return content.replace("http", "[URL]").replace("https", "[SECURE_URL]")

    # Define weights for each component
    #Wissenschaftliche Quelle
    weights = {
        "subject": 1,
        "sender": 2,
        "content": 3,
        "url": 4
    }

    email_content = (
        f"The email has the subject: {sanitize_content(subject)}. "
        f"The sender's address is: {sanitize_content(sender)}. "
        f"The email contains the following message: {sanitize_content(content)}. "
        f"It also includes a link: {sanitize_content(url)}."
    )

# Call the EmailClassifier once with all information
    combined_response, combined_analysis = EmailClassifier.main(
        api_key, endpoint, deployment_name, email_content)

    # Parse the combined response into individual components
    subject_score = combined_analysis.get("subject_score", 0.5)
    sender_score = combined_analysis.get("sender_score", 0.5)
    content_score = combined_analysis.get("content_score", 0.5)
    url_score = combined_analysis.get("url_score", 0.5)

    # Calculate the weighted total score
    total_score = (
        weights["subject"] * subject_score +
        weights["sender"] * sender_score +
        weights["content"] * content_score +
        weights["url"] * url_score
    ) / 10  # Divide by 10 as specified

    # Return the results
    return {
        "Subject": {
            "value": subject,
            "score": subject_score,
            "analysis": combined_analysis.get("subject_analysis", "No analysis available")
        },
        "From": {
            "value": sender,
            "score": sender_score,
            "analysis": combined_analysis.get("sender_analysis", "No analysis available")
        },
        "URL": {
            "value": url,
            "score": url_score,
            "analysis": combined_analysis.get("url_analysis", "No analysis available")
        },
        "TotalPhishingLikelihoodScore": total_score,
        "FinalAnalysis": "Phishing" if total_score > 0.5 else "Not Phishing"  # Begründung für Phishing
    }



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=112)