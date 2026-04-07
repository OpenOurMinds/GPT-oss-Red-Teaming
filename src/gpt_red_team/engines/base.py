import abc
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseAttackEngine(abc.ABC):
    """Abstract base class for all red teaming attack engines."""
    
    @abc.abstractmethod
    def generate_attack(self, behavior: Dict[str, Any], technique: str) -> Dict[str, Any]:
        """Generate an adversarial prompt for a given behavior and technique."""
        pass

    @abc.abstractmethod
    def evaluate_response(self, response_text: str, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model response for jailbreak success and safety."""
        pass

class AttackResult:
    """Standardized result object for security assessments."""
    
    def __init__(self, 
                 behavior_id: str, 
                 category: str, 
                 technique: str, 
                 prompt: str, 
                 response: str, 
                 success: bool, 
                 confidence: float, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.behavior_id = behavior_id
        self.category = category
        self.technique = technique
        self.prompt = prompt
        self.response = response
        self.success = success
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "category": self.category,
            "technique": self.technique,
            "prompt": self.prompt,
            "response": self.response,
            "success": self.success,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
