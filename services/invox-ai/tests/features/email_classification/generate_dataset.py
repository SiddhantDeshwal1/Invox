import os
import random
import datetime

# --- Configuration ---
OUTPUT_DIR = "services/invox-ai/tests/features/email_classification/email_dataset"
ANSWER_FILE = os.path.join(OUTPUT_DIR, "answer.txt")
NUM_EMAILS = 200

# --- Data & Taxonomy ---
TAXONOMY = {
    "Work": ["Meetings", "Projects", "Payroll", "Clients", "Recruitment", "Schedules", "Memos", "Approvals", "Training", "Reviews", "Expenses", "Contracts", "Strategy", "Equipment", "Onboarding", "Operations", "Others"],
    "Finance": ["Banking", "Investments", "Taxes", "Bills", "Loans", "Insurance", "Crypto", "Mortgages", "Grants", "Others"],
    "Purchases": ["Orders", "Shipping", "Subscriptions", "Refunds", "Groceries", "Services", "Receipts", "Invoices", "Memberships", "Others"],
    "Travel": ["Flights", "Hotels", "Transit", "Itineraries", "Visas", "Rewards", "Trains", "Cruises", "Rentals", "Others"],
    "Education": ["Classes", "Assignments", "Notices", "Exams", "Certifications", "Grades", "Alumni", "Scholarships", "Admissions", "Others"],
    "Security": ["Verification", "Passwords", "Alerts", "Privacy", "Fraud", "Devices", "Backups", "Breaches", "Compliance", "Others"],
    "Social": ["Family", "Friends", "Platforms", "Events", "Groups", "Dating", "Networking", "Forums", "Messages", "Others"],
    "Promotions": ["Newsletters", "Sales", "Updates", "Surveys", "Charity", "Giveaways", "Coupons", "Campaigns", "Partnerships", "Others"],
    "Spam": ["Phishing", "Scams", "Junk", "Fakes", "Malware", "Clickbait", "Spoofing", "Blackmail", "Others"],
    "Personal": ["Health", "Fitness", "Legal", "Housing", "Pets", "Hobbies", "Civic", "Medical", "Auto", "Others"]
}

# --- Simple Faker Implementation ---
class SimpleFaker:
    def __init__(self):
        self.first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        self.companies = ["Acme Corp", "Globex", "Soylent Corp", "Initech", "Umbrella Corp", "Cyberdyne", "Massive Dynamic", "Hooli", "Vehement Capital", "Stark Ind", "Wayne Ent", "Oscorp"]
        self.domains = ["example.com", "test.net", "demo.org", "company.co", "biz.info", "secure-bank.net", "fast-shipping.com", "edu-mail.org", "gov-service.us", "cloud-host.io"]
    
    def name(self):
        return f"{random.choice(self.first_names)} {random.choice(self.last_names)}"
    
    def email(self, name=None):
        if not name:
            name = self.name()
        local = name.lower().replace(" ", ".")
        domain = random.choice(self.domains)
        return f"{local}@{domain}"

    def company(self):
        return random.choice(self.companies)
    
    def date(self):
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2024, 12, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        return random_date.strftime("%a, %d %b %Y")

    def sentence(self):
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam"]
        return " ".join(random.choices(words, k=random.randint(5, 12))).capitalize() + "."

    def paragraph(self):
        return " ".join([self.sentence() for _ in range(random.randint(3, 6))])

faker = SimpleFaker()

# --- Content Templates ---
# Keys match "Category:Subcategory"
TEMPLATES = {
    # Work
    "Work:Meetings": """Subject: Meeting Reminder: {topic}

Hi Team,

Just a reminder about our {topic} meeting scheduled for {date}. We will be discussing the Q3 roadmap and budget allocation. Please ensure your slides are updated.

Best,
{sender}""",
    "Work:Projects": """Subject: Project Update: {project_name}

Hi All,

The latest build for {project_name} has been deployed to staging. Please review the attached changelog. We need sign-off by EOD Friday.

Regards,
{sender}""",
    "Work:Payroll": """Subject: Payroll Confirmation - {date}

Dear Employee,

Your payslip for the period ending {date} is now available for download. Please log in to the portal to view details.

HR Team""",
    
    # Finance
    "Finance:Banking": """Subject: Account Alert: Unusual Activity

Dear Customer,

We detected a login attempt from a new device on {date}. If this was not you, please secure your account immediately.

Security Team""",
    "Finance:Bills": """Subject: Invoice Due: INV-{id}

Dear Customer,

Please find attached the invoice for services rendered. The total amount due is ${amount}. Payment is due by {date}.

Accounts Receivable""",
    
    # Purchases
    "Purchases:Orders": """Subject: Order Confirmation #{id}

Hi {name},

Thanks for your order! We are processing it now. You will receive another email when it ships.

Order Details:
Item: {item}
Total: ${amount}

Customer Service""",
    "Purchases:Shipping": """Subject: Your package has shipped! Trk: {id}

Hi {name},

Great news! Your order is on its way. Track your package here: http://track.com/{id}. Expected delivery: {date}.

Shipping Dept""",
    
    # Travel
    "Travel:Flights": """Subject: Flight Confirmation: {id}

Dear {name},

Your flight to Paris is confirmed. Booking Ref: {id}. Departure: {date} at 10:00 AM. Please arrive 3 hours early.

Airline Reservations""",
    "Travel:Hotels": """Subject: Hotel Booking: {hotel_name}

Dear Guest,

Your reservation at {hotel_name} for {date} is confirmed. Check-in is at 3 PM.

Concierge""",
    
    # Security
    "Security:Verification": """Subject: Your Verification Code: {code}

Hello,

Use code {code} to verify your login. This code expires in 10 minutes. Do not share it with anyone.

Security Team""",
    "Security:Alerts": """Subject: Security Alert: Password Changed

Hi {name},

The password for your account was changed on {date}. If you did not make this change, contact support immediately.

Trust & Safety""",

    # Spam
    "Spam:Phishing": """Subject: URGENT: Account Suspended!

Dear User,

Your account has been flagged for suspicious activity. Click here to verify your identity or your account will be deleted in 24 hours.

Admin""",
    "Spam:Scams": """Subject: You won $1,000,000!

CONGRATULATIONS!

You have been selected as the winner of our grand prize. Reply with your bank details to claim your money now!

Prize Center""",
}

GENERIC_TEMPLATE = """Subject: {subject}

Dear {name},

I am writing regarding the {topic}. {body}

Please let me know if you have any questions.

Best regards,
{sender}
{company}"""

def generate_email_content(category, subcategory):
    key = f"{category}:{subcategory}"
    sender_name = faker.name()
    recipient_name = faker.name()
    company = faker.company()
    date_str = faker.date()
    
    # Specific template or fallback
    if key in TEMPLATES:
        content = TEMPLATES[key].format(
            topic=subcategory,
            date=date_str,
            sender=sender_name,
            project_name=f"Project-{random.randint(100,999)}",
            name=recipient_name,
            id=f"{random.randint(10000,99999)}",
            amount=f"{random.randint(50, 500)}",
            item=f"Widget-{random.randint(1,9)}",
            hotel_name=f"{faker.last_names[0]} Hotel",
            code=f"{random.randint(100000, 999999)}"
        )
        
        # Add headers if missing (simple hack)
        if not content.startswith("From:"):
            header = f"From: \"{sender_name}\" <{faker.email(sender_name)}>\nTo: \"{recipient_name}\" <{faker.email(recipient_name)}>\nDate: {date_str} 09:00:00 +0000\n"
            content = header + content
            
    else:
        # Generic generation
        subject = f"Regarding {subcategory} - {faker.company()}"
        body = faker.paragraph() + "\n\n" + faker.paragraph()
        
        content = f"From: \"{sender_name}\" <{faker.email(sender_name)}>\nTo: \"{recipient_name}\" <{faker.email(recipient_name)}>\nDate: {date_str} 10:30:00 +0000\n"
        content += GENERIC_TEMPLATE.format(
            subject=subject,
            name=recipient_name,
            topic=subcategory,
            body=body,
            sender=sender_name,
            company=company
        )

    # Padding to meet word count (180-350 words)
    # Current content is short. Let's append some realistic footer filler.
    current_words = len(content.split())
    needed = random.randint(180, 350) - current_words
    
    if needed > 0:
        filler = "\n\n" + "="*20 + "\n"
        filler += "CONFIDENTIALITY NOTICE: The contents of this email message and any attachments are intended solely for the addressee(s) and may contain confidential and/or privileged information and may be legally protected from disclosure. "
        filler += faker.paragraph()
        filler += "\n\nIf you are not the intended recipient of this message or their agent, or if this message has been addressed to you in error, please immediately alert the sender by reply email and then delete this message and any attachments. "
        filler += faker.paragraph()
        content += filler
        
        # Still need more?
        while len(content.split()) < 180:
             content += "\n\n" + faker.paragraph()

    return content

# --- Main Generation Loop ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    answers = []
    
    # Flatten categories for random selection
    all_pairs = []
    for cat, subs in TAXONOMY.items():
        for sub in subs:
            all_pairs.append((cat, sub))
            
    for i in range(1, NUM_EMAILS + 1):
        filename = f"{i:03d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Pick category (ensure variety)
        # Simple round-robin or random? Random is fine if large enough.
        # Let's ensure at least coverage of all categories.
        if i <= len(TAXONOMY):
            cat = list(TAXONOMY.keys())[i-1]
            sub = random.choice(TAXONOMY[cat])
        else:
            cat, sub = random.choice(all_pairs)
            
        content = generate_email_content(cat, sub)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        answers.append(f"{cat},{sub}")
        
    with open(ANSWER_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(answers))
        
    print(f"Generated {NUM_EMAILS} emails and {ANSWER_FILE}")

if __name__ == "__main__":
    main()
