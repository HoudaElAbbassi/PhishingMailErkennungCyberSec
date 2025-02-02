import re
import random


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


# Input email
email_text = "Subject: ebs research office space on 45 th or 43 rd floor  hi stinson , as per my discussion , here is e - mail to let vince know about the  subject .  we have requested the following office spaces for network planning ( jim  irvine ) - - optimization work , etc . and other ebs support personnel .  for network planning and traffic engineering :  - 1 director level space ( to be used by jim and other ebs research people  as needed when jim is not there ) and four other cube space ( chonowee , martin  lin , and two students .  for other ebs activities :  - 3 or 4 summer interns and analysts level space for other ebs people ( roman  and other students ) .  this is what kristy is going to request and get it into the office space  request loop . we should be able to get into most of the space that i have  requested .  ravi .,0"

# Extract details
email_details = extract_email_details(email_text)

# Print extracted details
for key, value in email_details.items():
    print(f"{key}: {value}")
