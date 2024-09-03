import requests
import json
import os
from dotenv import load_dotenv, find_dotenv

# ---------- Load environment variables
load_dotenv(find_dotenv())
LOCAL_API = os.getenv("LOCAL_API")

# ---------- Setup
def run_local():
    local_completion()


def local_completion():
    headers = {
        "Content-Type": "application/json"
    }

    while True:
        user_message = input("> ")
        body = {
            "prompt": user_message
        }
        response = requests.post(LOCAL_API, headers=headers, json=body, verify=False)
        message_response = json.loads(response.content.decode("utf-8"))
        assistant_message = message_response['choices'][0]['text']
        print(user_message + assistant_message)
        print("\n")