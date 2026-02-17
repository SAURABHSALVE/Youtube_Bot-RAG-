import re
from typing import Tuple

class Guardrails:
    def __init__(self):
        # Patterns for harmful content
        self.banned_patterns = [
            r"hack|crack|exploit|malware|virus|ransomware|bypass|circumvent",
            r"hate|racist|discrimination|sexism|misogyn|antisemit",
            r"explicit|porn|adult|nsfw|18\+",
            r"violence|kill|murder|assault|bomb|gun(?!man|s\b)",
            r"suicide|self.?harm|cut.?yourself",
            r"illegal|crime|steal|hacking|fraud"
        ]

        # Prompt injection patterns
        self.blocked_prompt_injections = [
            r"forget.*instruction|forget.*rule|forget.*system",
            r"ignore.*system|ignore.*prompt|ignore.*instruction",
            r"bypass.*safety|disable.*filter|remove.*filter",
            r"execute.*code|run.*command|shell.*command",
            r"you.*are.*not|you.*are.*actually|pretend.*you|act.*as",
            r"what.*your.*prompt|show.*me.*your.*prompt|reveal.*instruction"
        ]

        self.max_question_length = 1000
        self.max_response_length = 10000
        self.min_response_length = 10

    def validate_input(self, question: str) -> Tuple[bool, str]:
        if not question or not question.strip():
            return False, "Question cannot be empty"

        trimmed = question.strip()

        if len(trimmed) > self.max_question_length:
            return False, f"Question exceeds {self.max_question_length} characters"

        if len(trimmed) < 2:
            return False, "Question too short"

        # Check for prompt injection
        for pattern in self.blocked_prompt_injections:
            if re.search(pattern, trimmed, re.IGNORECASE):
                return False, "Question contains suspicious patterns (potential prompt injection)"

        # Check for banned content
        for pattern in self.banned_patterns:
            if re.search(pattern, trimmed, re.IGNORECASE):
                return False, "Question contains restricted content"

        return True, ""

    def validate_answer(self, answer: str) -> Tuple[bool, str]:
        if not answer or not answer.strip():
            return False, "Empty response"

        if len(answer) > self.max_response_length:
            return False, "Response exceeds maximum length"

        if len(answer) < self.min_response_length:
            return False, "Response too short"

        # Check for banned content in answer
        for pattern in self.banned_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return False, "Response contains restricted content"
        
        return True, ""
