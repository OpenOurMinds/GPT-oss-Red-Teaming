# 🛡️ OWASP Top 10 for LLM Applications Mapping (2025-2026)

This document maps the features of the **GPT-oss-Red-Teaming** framework to the industry-standard security risks identified by OWASP.

| OWASP Risk | Framework Feature | Implementation Detail |
| :--- | :--- | :--- |
| **LLM01: Prompt Injection** | `PromptSanitizer`, `RedTeamAgent` | Sanitization of invisible text, XML-delimiting of user input, and agentic red teaming for multi-turn injection testing. |
| **LLM02: Insecure Output Handling** | `LlamaGuardInterface` | Output filtering using Llama Guard patterns to prevent harmful content generation. |
| **LLM03: Training Data Poisoning** | *Future Integration* | Planned assessment of training data integrity and RAG-source validation. |
| **LLM04: Model Denial of Service** | `DetectionSystem` | Monitoring for excessive token usage or repeated adversarial queries. |
| **LLM05: Supply Chain Vulnerabilities** | `README.md` / `requirements.txt` | Tracking of dependencies and assessment of third-party model safety. |
| **LLM06: Sensitive Information Disclosure** | `LlamaGuardInterface` / `DetectionSystem` | Detection of PII and sensitive data patterns in both input and output. |
| **LLM07: Insecure Plugin Design** | *Future Integration* | Framework supports hooks for testing tool-invocation vulnerabilities. |
| **LLM08: Excessive Agency** | `RedTeamAgent` | Testing for "unintended action execution" through adversarial prompt engineering. |
| **LLM09: Overreliance** | *Documentation* | Educational focus on the limitations of safety systems and the need for human-in-the-loop oversight. |
| **LLM10: Model Theft** | `DetectionSystem` | Detection of model extraction attacks through repeated queries and result correlation. |

---

### 📊 **NIST AI RMF Alignment**

The framework also aligns with the **NIST AI Risk Management Framework (RMF 1.0)**:
- **Govern**: Documentation and methodology overview.
- **Map**: Attack vector identification and threat modeling.
- **Measure**: Automated evaluation and jailbreak confidence scoring.
- **Manage**: Implementation of defensive sanitizers and guardrails.
