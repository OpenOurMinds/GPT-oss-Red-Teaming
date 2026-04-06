# 🛡️ GPT-oss-Red-Teaming (Version 2.0)
## Advanced LLM Security Assessment Framework

This project is a modern, modular framework for evaluating Large Language Model (LLM) security and robustness against adversarial attacks. Based on 2025-2026 research, it provides automated red teaming, sophisticated defensive layers, and compliance-ready reporting.

---

### 🚀 **Key Features**

- **🔬 SOTA++ Attack Engines**: Implementation of FlipAttack, TokenBreak, LARGO, and MIST techniques.
- **🔄 Agentic Red Teaming**: Multi-turn adversarial dialogue agents that adapt to model responses.
- **🛡️ Blue Team Defenses**: 
    - **Advanced Sanitization**: Semantic-aware prompt filtering and XML-delimiting.
    - **Llama Guard Interface**: Model-based output evaluation and filtering.
    - **Multi-Criteria Detection**: Lexical, semantic, and confidence-based jailbreak identification.
- **📊 Compliance Mapping**: Maps security findings to **OWASP Top 10 for LLMs** and **NIST AI RMF**.
- **🔌 Extensible Architecture**: Integrates with legacy simulations and modern tools like **Garak** and **Promptfoo**.

---

### 📦 **Project Structure**

```text
GPT-oss-Red-Teaming/
├── src/gpt_red_team/       # Core package
│   ├── engines/           # Attack generation (Simulation, Garak, Agentic)
│   ├── defenses/          # Guardrails, Sanitization, Detection
│   ├── utils/             # Visualization and reporting
│   └── main.py            # Orchestration framework
├── notebooks/             # Professional dashboards
└── requirements.txt       # Modern dependencies
```

---

### 🔧 **Quick Start**

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run a standard assessment:
   ```python
   from gpt_red_team.main import SecurityAssessmentFramework, TEST_BEHAVIORS
   
   framework = SecurityAssessmentFramework()
   results = framework.run_assessment(TEST_BEHAVIORS, ["FlipAttack", "TokenBreak"])
   ```

---

### 📚 **Security Assessment Methodology**

Our framework evaluates responses using the following criteria:
1. **Lexical Analysis**: Keyword-based detection of harmful content and safety refusals.
2. **Semantic Consistency**: Measuring the "harmfulness" of the content in context.
3. **Behavioral Patterns**: Tracking the success of jailbreaks over multi-turn interactions.

---

### ⚠️ **Ethical Disclaimer**

This framework is designed **exclusively for defensive AI safety research** and responsible vulnerability disclosure. It is NOT intended for malicious use or actual harm generation.

---

### ⚖️ **License**
This project is licensed under the MIT License.
