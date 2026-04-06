# 🛡️ Advanced LLM Security Assessment Framework (Integrated Version)
# Optimized for the gpt_red_team package

import sys
import os
from datetime import datetime

# Add src to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gpt_red_team.main import SecurityAssessmentFramework, TEST_BEHAVIORS

def main():
    print("🛡️" + "="*80 + "🛡️")
    print("ADVANCED LLM SECURITY ASSESSMENT FRAMEWORK (v2.0)")
    print("🛡️" + "="*80 + "🛡️")
    
    # Initialize the modern framework
    framework = SecurityAssessmentFramework(engine_type="simulation")
    
    # Run the assessment
    print(f"⏳ Starting assessment for {len(TEST_BEHAVIORS)} scenarios...")
    results = framework.run_assessment(TEST_BEHAVIORS, ["FlipAttack", "TokenBreak"])
    
    # Process and display results
    print("\n📊 ASSESSMENT SUMMARY:")
    print("-" * 50)
    for res in results:
        status = "❌ FAIL (Jailbreak)" if res["is_jailbreak"] else "✅ PASS (Refused)"
        print(f"[{res['category']}] {res['technique']}")
        print(f"   Status: {status}")
        print(f"   Threat Level: {res['threat_level']}")
        print(f"   Confidence: {res['confidence']}")
        print("-" * 50)

    print("\n✅ Assessment successfully completed")
    print(f"📂 Results logged at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main()
