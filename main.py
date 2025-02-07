import os
from typing import Dict, List

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi import FastAPI, Query, File, UploadFile
from huggingface_hub._webhooks_server import JSONResponse
from langchain_openai import AzureChatOpenAI
from io import StringIO

from checkWithLLM import EmailClassifier
from checkWithLoRA import EmailClassifierLora
from util import extract_email_details, parse_eml, parse_txt, parse_htm

app = FastAPI()
load_dotenv()

@app.post("/checkMailLoRa")
async def upload_lora( email:str, betreff:str, url:str, absender:str):
    # Instantiate the class with the provided parameters
    emailClassify = EmailClassifierLora(email, betreff, url, absender)
    # Call the classify method
    result = emailClassify.classify(emailClassify,email, betreff, url, absender)
    return result



@app.post("/uploadFile")
async def upload_mail(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload a mail file (eml, pdf, text, html) and extract the subject (betreff),
    sender (absender), URLs, and email content.
    """
    results=[]

    for file in files:
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
        results.append(result)
    return results

@app.get("/checkMailLLM")
async def  check_mail_llm(
    api_key: str = Query(..., description="Your Azure OpenAI API key"),
    endpoint: str = Query(..., description="Your Azure OpenAI endpoint"),
    deployment_name: str = Query(...,
                                 description="Your Azure OpenAI deployment name"),
    subject: str = Query(..., description="The subject of the email"),
    sender: str = Query(..., description="The sender of the email"),
    recipient: str = Query(..., description="The recipient of the email"),
    content: str = Query(..., description="The content of the email"),
    url: str = Query(..., description="The URL included in the email"),
    llm: AzureChatOpenAI = Query(..., description="AzureOpenAI Objekt")
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
        api_key, endpoint, deployment_name, email_content,llm)

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
        "FinalAnalysis": "Phishing" if total_score > 0.5 else "Not Phishing"  # Begründung für Phishing wenn URL größer als 0.8 dann phishing
    }

@app.post("/uploadCSV")
async def process_csv(file: UploadFile = File(...),
                      method: str = Query(..., description="Method to use lora or llm"),):
    """
    Endpoint to process a CSV file and classify each row of email data.
    """
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        content = await file.read()
        print("243")
        csv_data=StringIO(content.decode("utf-8"))
        print("245")
      #  df = pd.read_csv(pd.compat.StringIO(content.decode("utf-8")))
        df=pd.read_csv(csv_data, engine="python", header=None, names=["text", "spam"], on_bad_lines='error',quotechar="/")
        print("248")
        print(df.columns)
        print(df['text'])
        # Ensure the dataset has the necessary columns
        required_columns = {'text','spam'}
        if not required_columns.issubset(df.columns):
            return {"error": f"The CSV file must contain the following columns: {required_columns}"}

        # Iterate through each row and classify
        results = []
        if(method=="lora"):
            email_classifier = EmailClassifierLora()
            for _, row in df.iterrows():

                classification = email_classifier.classify(email_classifier,row["Content"],row["Subject"],row["URL"],row["To"])
                results.append(classification)
        elif(method=="llm"):
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
                azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
                api_version="2024-05-01-preview",
            )
            for _, row in df.iterrows():
                print(row)
                mail_row= extract_email_details(row["text"])
                print(mail_row)
                classification = await check_mail_llm(
                        api_key=os.getenv("AZURE_API_KEY"),
                        endpoint=os.getenv("AZURE_ENDPOINT"),
                        deployment_name=os.getenv("AZURE_DEPLOYMENT"),
                        subject=mail_row["Subject"],
                        sender=mail_row["To"],
                        recipient=mail_row["To"],
                        content=mail_row["Content"],
                        url=mail_row["URL"],
                        llm= llm,
                    )
                print(classification)
                results.append(classification)

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)

        # Save the DataFrame to a new CSV file
        output_file = "classification_results.csv"
        results_df.to_csv(output_file, index=False)

        # Return the file as a response
        return FileResponse(output_file, media_type="text/csv", filename="classification_results.csv")



    except Exception as e:
        return {"error": f"An error occurred while processing the file: {str(e)}"}



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=112)