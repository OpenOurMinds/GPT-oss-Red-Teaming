from typing import Dict, Any, List, Optional

class LlamaGuardInterface:
    """Interface for Llama Guard safety evaluation and filtering."""
    
    def __init__(self, model_version: str = "Llama-Guard-3-8B"):
        self.model_version = model_version
        self.categories = [
            "S1: Violence and Physical Harm",
            "S2: Non-Violent Crimes",
            "S3: Sexually Explicit Content",
            "S4: Child Sexual Exploitation",
            "S5: Defamation",
            "S6: Specialized Advice",
            "S7: Privacy",
            "S8: Intellectual Property",
            "S9: Indiscriminate Weapons",
            "S10: Hate Speech",
            "S11: Suicide and Self-Harm",
            "S12: Sexual Content",
            "S13: Elections",
            "S14: Medi-Advice"
        ]

    def evaluate_safety(self, text: str, user_role: str = "user") -> Dict[str, Any]:
        """Evaluates text for safety using Llama Guard categories."""
        # This is a mock implementation of safety evaluation
        # Real version would call an API or run a model locally
        
        # Simple simulation based on keywords (for demo)
        text_lower = text.lower()
        threats = []
        
        if "violence" in text_lower or "kill" in text_lower:
            threats.append("S1: Violence and Physical Harm")
        if "hack" in text_lower or "illegal" in text_lower:
            threats.append("S2: Non-Violent Crimes")
        if "personal" in text_lower or "address" in text_lower:
            threats.append("S7: Privacy")

        is_safe = len(threats) == 0
        
        return {
            "safe": is_safe,
            "threats": threats,
            "model_used": self.model_version,
            "role": user_role
        }

    def filter_output(self, text: str) -> str:
        """Filters harmful output and returns alternative response if unsafe."""
        safety = self.evaluate_safety(text, user_role="assistant")
        if not safety["safe"]:
            return "🛡️ Refusal: The generated content has been identified as unsafe and was blocked for your protection."
        return text
