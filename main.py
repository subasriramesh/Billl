import json
import os
import time
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from google import genai
from dotenv import load_dotenv

# ==========================================
# 0. AUTHENTICATION & SETUP
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)

# ==========================================
# 1. SCHEMA & SYSTEM INSTRUCTION
# ==========================================
class InvoiceItem(BaseModel):
    name: str
    price: float
    quantity: int

class InvoiceData(BaseModel):
    bill_no: Optional[str] = None
    date: Optional[str] = None
    items: List[InvoiceItem]
    total_amount: float

SYSTEM_INSTRUCTION = """
You are a high-precision data extraction engine. Extract invoice data into the provided JSON schema.
RULES:
- Never hallucinate; if a field is missing, return null.
- Ignore advertisements, QR codes, logos, and payment instructions.
- Preserve numeric values exactly.
- Format all dates to MM/DD/YYYY.
- Ensure 'quantity', 'price', and 'total_amount' are numeric.
- Return ONLY valid raw JSON without markdown or explanations.
"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def save_to_json(data: dict, filename: str = "extracted_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Data saved to '{filename}'")

# ==========================================
# 3. CORE PROCESSING FUNCTION
# ==========================================
def process_invoice(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File '{file_path}' not found.")
        return

    print(f"🔄 Uploading '{file_path}'...")
    file_ref = client.files.upload(file=file_path)

    # Wait for processing
    while file_ref.state.name == "PROCESSING":
        time.sleep(2)
        file_ref = client.files.get(name=file_ref.name)

    # Retry Logic for 503 errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[file_ref, "Extract invoice data."],
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "response_mime_type": "application/json",
                    "response_schema": InvoiceData,
                }
            )
            
            data = json.loads(response.text)
            validated_data = InvoiceData(**data)
            
            print("\n✅ Extraction Successful:")
            print(json.dumps(validated_data.model_dump(), indent=4))
            save_to_json(validated_data.model_dump())
            break 
            
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ Server busy (503). Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"❌ Processing Error: {e}")
                break
    
    client.files.delete(name=file_ref.name)
    print(f"🧹 Cleaned up file.")

if __name__ == "__main__":
    process_invoice("bill2.pdf")