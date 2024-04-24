from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
prompt_text = """
Write me some reviews that imply what time of the day the reviewer visited a business.Give me as much reviews as you can.
Please also add the time of day the event happens after generated the reveiew.
"""

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt_text,
        }
    ],
    model="gpt-3.5-turbo",
)
# Example prompts for different types of reviews

print(chat_completion.choices[0].message.content)

response_text = chat_completion.choices[0].message.content

with open("./data/responses.txt", "a") as file:
    file.write(response_text + "\n")
    
