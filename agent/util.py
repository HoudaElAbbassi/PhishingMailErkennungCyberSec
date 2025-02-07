import re
import random
from email import message_from_bytes
from typing import Dict

from bs4 import BeautifulSoup
from requests.compat import chardet


def generate_random_email():
    domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com"]
    user = "user" + str(random.randint(1000, 9999))
    return f"{user}@{random.choice(domains)}"


def extract_email_details(email_text):
    # Define regex patterns
    subject_pattern = r"Subject:\s*(.*?)\n"
    url_pattern = r"http\s*:\s*\/\s*\/\s*[\w\.\/]+"
    from_pattern = r"From:\s*(.*?)\n"

    # Extract subject
    subject_match = re.search(subject_pattern, email_text, re.IGNORECASE)
    subject = subject_match.group(1) if subject_match else ""

    # Extract URL and clean it
    url_match = re.search(url_pattern, email_text)
    url = url_match.group(0) if url_match else ""
    url = re.sub(r'\s*:\s*', ':', url)
    url = re.sub(r'\s*\/\s*', '/', url)
    url = re.sub(r'\s+', '', url)

    # Extract 'From' field if present
    from_match = re.search(from_pattern, email_text, re.IGNORECASE)
    sender = from_match.group(1) if from_match else generate_random_email()

    # Extract 'To' field if present
    to_pattern = r"To:\s*(.*?)\n"
    to_match = re.search(to_pattern, email_text, re.IGNORECASE)
    to = to_match.group(1) if to_match else "Unknown"

    # Extract content by removing subject, URL, From, and To fields
    content = email_text
    if subject:
        content = content.replace(f"Subject: {subject}", "").strip()
    if url:
        content = content.replace(url_match.group(0), "").strip()
    if sender:
        content = content.replace(f"From: {sender}", "").strip()
    if to != "Unknown":
        content = content.replace(f"To: {to}", "").strip()

    return {
        "Subject": subject,
        "Content": content,
        "URL": url,
        "From": sender,
        "To": sender
    }

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


# Input email
email_text = "Subject: ebs research office space on 45 th or 43 rd floor  hi stinson , as per my discussion , here is e - mail to let vince know about the  subject .  we have requested the following office spaces for network planning ( jim  irvine ) - - optimization work , etc . and other ebs support personnel .  for network planning and traffic engineering :  - 1 director level space ( to be used by jim and other ebs research people  as needed when jim is not there ) and four other cube space ( chonowee , martin  lin , and two students .  for other ebs activities :  - 3 or 4 summer interns and analysts level space for other ebs people ( roman  and other students ) .  this is what kristy is going to request and get it into the office space  request loop . we should be able to get into most of the space that i have  requested .  ravi .,0"

# Extract details
email_details = extract_email_details(email_text)

# Print extracted details
for key, value in email_details.items():
    print(f"{key}: {value}")
