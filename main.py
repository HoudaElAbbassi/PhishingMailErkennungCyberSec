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
async def check_mail(api_key, endpoint, deployment_name, prompt, email):
    return EmailClassifier(api_key, endpoint, deployment_name, prompt, email).main(api_key, endpoint, deployment_name, prompt, email)


@app.get("/checkMailLoRa")
async def check_mail_lora(mail: str, betreff: str):
    return LocalModelEmailClassifier("/Users/houda/Downloads/lora_finetuned_model/").main()
