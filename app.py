from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import gradio as gr

# Load environment variables (works for both local .env and Hugging Face Spaces)
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

# Validate required environment variables
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

welcome_message = os.getenv("WELCOME_MESSAGE", "Welcome message not set.")
print(f"Loaded WELCOME_MESSAGE: {welcome_message}")
system_prompt_text = os.getenv("SYSTEM_PROMPT_TEXT", "System prompt not set.")
print(f"Loaded SYSTEM_PROMPT_TEXT: {system_prompt_text}")

def push(text):
    try:
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_user = os.getenv("PUSHOVER_USER")
        
        if pushover_token and pushover_user:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": pushover_token,
                    "user": pushover_user,
                    "message": text,
                }
            )
        else:
            print(f"Pushover notification (not configured): {text}")
    except Exception as e:
        print(f"Failed to send pushover notification: {e}")
        print(f"Message was: {text}")


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Subir Roy"
        
        # Load LinkedIn profile from environment variable
        linkedin_text = os.getenv('LINKEDIN_PROFILE_TEXT')
        if linkedin_text:
            self.linkedin = linkedin_text
            print("LinkedIn profile loaded from environment variable")
            print(f"Loaded LINKEDIN_PROFILE_TEXT: {linkedin_text}")
        else:
            print("Warning: LINKEDIN_PROFILE_TEXT environment variable not set")
            self.linkedin = "LinkedIn profile information not available. Please set LINKEDIN_PROFILE_TEXT environment variable."
        
        # Load summary from environment variable
        summary_text = os.getenv('PROFESSIONAL_SUMMARY')
        if summary_text:
            self.summary = summary_text
            print("Professional summary loaded from environment variable")
            print(f"Loaded PROFESSIONAL_SUMMARY: {summary_text}")
        else:
            print("Warning: PROFESSIONAL_SUMMARY environment variable not set")
            self.summary = "Professional summary not available. Please set PROFESSIONAL_SUMMARY environment variable."


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = system_prompt_text.format(name=self.name)
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    
    # Rollback: Use the original approach with gr.Chatbot for compatibility with Gradio 5.34.2
    demo = gr.ChatInterface(
        me.chat,
        chatbot=gr.Chatbot(value=[{"role": "assistant", "content": welcome_message}], type="messages"),
        title="Career Conversation with Subir Roy",
        description="Chat with an AI representation of Subir Roy about his professional career, background, and experience.",
        examples=[
            "Tell me about your professional background",
            "What are your key skills and expertise?",
            "What industries have you worked in?",
            "Can you share your career journey?",
            "What are your current professional interests?"
        ],
        theme=gr.themes.Soft(),
        type="messages"
    )
    
    # Launch for local development or Hugging Face Spaces
    demo.launch(inbrowser=True)
    