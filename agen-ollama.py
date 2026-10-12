
import json
import os
import subprocess
from openai import OpenAI

# Ollama's OpenAI-compatible local API
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Local Ollama does not require a real API key
)

MODEL = "llama3.2:latest"

SYSTEM_PROMPT = """
You are a coding agent running in the user's terminal.

You can:
- List files and directories.
- Read text files.
- Create or overwrite text files.
- Run shell commands after user approval.

The working directory is the folder from which the agent was launched.
Use your tools whenever necessary.
Never claim a file was changed or a command succeeded unless the tool confirms it.
Ask for approval before running shell commands.
Keep responses concise.
"""


# --------------------------------------------------
# Tool implementations
# --------------------------------------------------

def list_files(path="."):
    entries = []

    for entry in os.scandir(path):
        entries.append(
            entry.name + ("/" if entry.is_dir() else "")
        )

    return "\n".join(sorted(entries)) or "(empty directory)"


def read_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def write_file(path, content):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

    return f"Saved {path} ({len(content)} characters)"


def run_command(command):
    print(f"\nCommand requested:\n{command}")

    answer = input("Run this command? [y/N]: ")

    if answer.strip().lower() != "y":
        return "The user declined to run this command."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
        )

        output = (result.stdout + result.stderr).strip()

        return (
            f"Exit code: {result.returncode}\n"
            f"Output:\n{output or '(no output)'}"
        )

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


# --------------------------------------------------
# OpenAI-compatible tool schemas for Ollama
# --------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path, such as .",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file contents.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command after user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


# --------------------------------------------------
# Execute model-requested tools
# --------------------------------------------------

def run_tool(tool_call):
    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments)

        print(f"\n[Tool] {name}: {args}")

        if name not in TOOLS:
            return f"Error: Unknown tool '{name}'."

        return str(TOOLS[name](**args))

    except Exception as error:
        return f"Error executing tool: {error}"


# --------------------------------------------------
# Agent loop
# --------------------------------------------------

def run_agent(messages):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        # Preserve the assistant's tool-call structure.
        messages.append(
            message.model_dump(exclude_none=True)
        )

        # No tool calls: return the final answer.
        if not message.tool_calls:
            return message.content or "(No response generated.)"

        # Execute every requested tool and send results back.
        for tool_call in message.tool_calls:
            result = run_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


# --------------------------------------------------
# Interactive terminal
# --------------------------------------------------

def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Local Coding Agent")
    print(f"Model: {MODEL}")
    print("Provider: Ollama")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("Agent stopped.")
                break

            if not user_input:
                continue

            messages.append(
                {"role": "user", "content": user_input}
            )

            try:
                reply = run_agent(messages)
                print(f"\nAgent: {reply}")

            except Exception as error:
                print(f"\nAgent error: {error}")
                print(
                    "Check that Ollama is running, the model "
                    "is installed, and tool calling is supported."
                )

        except (KeyboardInterrupt, EOFError):
            print("\nAgent stopped.")
            break


if __name__ == "__main__":
    main()
