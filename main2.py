import json
import os
import time
from typing import List, Optional, Union
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# ==========================================
# 0. AUTHENTICATION & SETUP
# ==========================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# 1. UPDATED SCHEMA
# ==========================================
class InvoiceItem(BaseModel):
    name: str
    price: float
    quantity: int

class InvoiceData(BaseModel):
    bill_no: Optional[str] = None
    date: Optional[str] = None
    items: List[InvoiceItem]
    # We use Union to allow either a number (if found) or a string (for the 'no tax' message)
    tax_amount: Union[float, str] = "No tax is filled"
    total_amount: float

SYSTEM_INSTRUCTION = """
You are a high-precision data extraction engine. Extract invoice data into the provided JSON schema.
RULES:
- If an explicit tax amount is found, return it as a number (float).
- If no tax amount is mentioned or the bill does not show a tax breakdown, return the exact string: "No tax is filled".
- Do not perform any calculations. Extract exactly what is written on the document.
- Ignore advertisements, QR codes, logos, and payment instructions.
- Preserve numeric values exactly.
- Format dates to MM/DD/YYYY.
- Return ONLY valid raw JSON without markdown or explanations.
"""

# ==========================================
# 2. PROCESSING FUNCTION
# ==========================================
def process_invoice(file_path: str):
    print(f"🔄 Uploading '{file_path}'...")
    file_ref = client.files.upload(file=file_path)

    while file_ref.state.name == "PROCESSING":
        time.sleep(2)
        file_ref = client.files.get(name=file_ref.name)

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
        
        # Parse, Validate, and Save
        data = json.loads(response.text)
        validated_data = InvoiceData(**data)
        
        output_dict = validated_data.model_dump()
        
        print("\n✅ Extraction Successful:")
        print(json.dumps(output_dict, indent=4))
        
        with open("extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)
        print("\n💾 Data saved to 'extracted_data.json'")
            
    finally:
        client.files.delete(name=file_ref.name)

if __name__ == "__main__":
    process_invoice("bill3.pdf")