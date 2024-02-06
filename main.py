from openai import OpenAI
import os
from dotenv import load_dotenv
from messages import SYSTEM_MSG, USER_MSG

load_dotenv()
      
client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

conversation = [
        {
            "role" : "system",
            "content" : SYSTEM_MSG
        },
        {
            "role": "user",
            "content" : USER_MSG
        }

    ]

completion = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = conversation,
)



print(completion.choices[0].message.content)