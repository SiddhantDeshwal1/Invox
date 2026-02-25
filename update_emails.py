import os
import re
import random

output_dir = "services/invox-ai/tests/features/email_classification/email_dataset/"

def generate_random_attachment():
    prefix = random.choice(["Document", "Report", "Summary", "Details", "Invoice_Draft", "Brief", "Scan"])
    num = random.randint(1000, 9999)
    ext = random.choice([".pdf", ".docx", ".xlsx"])
    return f"Attachment: {prefix}_{num}{ext}"

files = [f for f in os.listdir(output_dir) if f.endswith(".txt") and f != "answer.txt"]

for filename in files:
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace Ref: CATEGORY-SUBCATEGORY-XXX
    new_content = re.sub(r"Ref: [A-Z]+-[A-Z]+-\d+", lambda m: generate_random_attachment(), content)
    
    # Also replace explicit metadata that might be too revealing if it contains category names
    # (Though in the previous script, most were Order ID, Invoice, etc. which are fine)
    
    with open(filepath, 'w') as f:
        f.write(new_content)

print(f"Updated {len(files)} files.")
