import json

from openai import OpenAI

client = OpenAI(api_key="")



def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {path} not found"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read"},
                },
                "required": ["path"],
            },
        },
    },
]

messages = [
    {"role": "user", "content": "What is inside notes.txt, notes1.txt and notes2.txt? Summarize it in one line."},
]

while True:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=TOOL_SCHEMAS,
    )
    message = response.choices[0].message
    messages.append(message)

    # No tool calls means the model is done and gave us a normal answer
    if not message.tool_calls:
        print(message.content)
        break

    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print(f"Model wants to run: read_file({args})")

        result = read_file(**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
    
    print(messages)
