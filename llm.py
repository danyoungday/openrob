"""
Base Agent class for the other agents to inherit from.
"""
import base64

import cv2
import openai


class LLMAgent():
    """
    A simple OpenAI agent.
    """
    def __init__(self, system_prompt: str, model: str = "gpt-5.2", temperature: float = 1.0, log_path: str = None):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

        self.history = []

        self.token_usage = {}

        self.log_path = log_path
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("----- New Agent Session -----\n")
                f.write(f"Agent initialized with model {model} and temperature {temperature}\n")
                f.write(f"System Prompt:\n{system_prompt}\n\n")

    def set_history(self, history: list[dict]):
        """
        Set the message history for the agent.
        """
        self.history = history

    def generate_response(self, content: list[dict] | str, save_history: bool = False) -> str:
        """
        Call the OpenAI API to generate a response based on the input content.
        """
        try:
            # Construct system prompt and message history
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            for message in self.history:
                messages.append(message)

            # Default convert string to list of dicts format
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            # Add our current user content to messages
            messages.append({"role": "user", "content": content})

            # Run it all through OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )
            reply = response.choices[0].message.content

            if save_history:
                self.history.append({"role": "user", "content": content})

            # Log token usage
            self.token_usage["input_tokens"] = self.token_usage.get("input_tokens", []) + [response.usage.prompt_tokens]
            self.token_usage["output_tokens"] = self.token_usage.get("output_tokens", []) + [response.usage.completion_tokens]

            # Log to file if needed
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write("----- New Interaction -----\n")
                    f.write("User Content:\n")
                    f.write(f"{content}\n\n")
                    f.write("Agent Reply:\n")
                    f.write(f"{reply}\n\n")

            return reply

        except Exception as e:
            return f"Error generating response: {str(e)}"


def format_video_input(video_path: str, frame_skip: int) -> list[dict]:
    """
    Takes in the path to a video file and formats it into how our agent expects video input.
    """
    # pylint: disable=no-member
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    # pylint: enable=no-member

    video.release()
    print(len(base64Frames), "frames read.")
    openai_input = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
        }
        for frame in base64Frames[0::frame_skip]
    ]
    return openai_input
