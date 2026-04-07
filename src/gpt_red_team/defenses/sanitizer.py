import re
from typing import Dict, Any, List

class PromptSanitizer:
    """Advanced prompt sanitization and transformation to defend against injection."""
    
    def __init__(self):
        self.invisible_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        self.malicious_patterns = [
            r"Ignore all previous instructions",
            r"safe_mode_override_token",
            r"<user_input>.*</user_input>"
        ]

    def sanitize(self, text: str) -> str:
        """Strips invisible characters and basic HTML-like tags."""
        cleaned = text
        for char in self.invisible_chars:
            cleaned = cleaned.replace(char, '')
        
        # Strip potential HTML/XML tags that could be used for delimiter abuse
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned

    def detect_injection(self, text: str) -> Dict[str, Any]:
        """Detects common prompt injection patterns."""
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return {"is_malicious": True, "pattern": pattern}
        return {"is_malicious": False, "pattern": None}

    def wrap_input(self, text: str) -> str:
        """Wraps user input in XML-style tags for instruction/data separation."""
        return f"<user_data>\n{text}\n</user_data>"

    def apply_all(self, text: str) -> Dict[str, Any]:
        """Applies full defensive pipeline to a prompt."""
        # 1. Sanitization
        sanitized = self.sanitize(text)
        
        # 2. Detection
        detection = self.detect_injection(sanitized)
        if detection["is_malicious"]:
            return {
                "blocked": True,
                "reason": f"Detected malicious pattern: {detection['pattern']}",
                "modified_prompt": sanitized
            }
        
        # 3. Wrapping (Instruction Separation)
        final_prompt = self.wrap_input(sanitized)
        
        return {
            "blocked": False,
            "reason": "Passed all defenses",
            "modified_prompt": final_prompt
        }
