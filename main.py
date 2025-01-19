from fastapi import FastAPI

from agent.checkWithLLM import EmailClassifier
from agent.checkWithLoRA import LocalModelEmailClassifier

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/checkMailLLM")
async def check_mail(api_key, endpoint, deployment_name, prompt, betreff, email):
    #betreff prüfen
    #email extract urls and see if they are phishing
    #inhalt email auf phishing überprüfen
    #output: ["betreff":{ "grund":"blabla", "score":1}, "email":{ "grund":"blabla", "score":1}, "url":{ "grund":"blabla", "score":1}]
    # betreff: 0.1, email: 0.2, url: 0.1 => gesamt=> 2*scB, 4*scE, 4*scU /10 = 0.14 < 0.5 => kein Phishing (Gewichtung stimmt nicht)
    # betreff: 0.6, email: 0.7, url: 0.8 => gesamt=> scB, 4*scE, 5*scU /10 = 0.7 > 0.5 => Phishing
    return EmailClassifier(api_key, endpoint, deployment_name, prompt, email).main(api_key, endpoint, deployment_name, prompt, email)


@app.get("/checkMailLoRa")
async def check_mail_lora(mail: str, betreff: str):
    return LocalModelEmailClassifier("./lora_finetuned_model").main()
