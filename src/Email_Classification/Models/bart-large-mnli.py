import sys
from transformers import pipeline


def classify(file_path):
    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli", device=-1
    )

    labels = [
        "Job Offer & Contract",
        "Interview Invitation",
        "Job Application Rejection",
        "Recruiter Outreach",
        "Freelance Gig Inquiry",
        "Salary Negotiation",
        "Internal Team Update",
        "Meeting Agenda & Minutes",
        "Project Management Alert",
        "IT Support Ticket",
        "Corporate Policy Change",
        "Human Resources & Benefits",
        "Bank Account Statement",
        "Investment Portfolio Report",
        "Dividend & Interest Alert",
        "Cryptocurrency Transaction",
        "Tax Filing Document",
        "Inward Remittance & Wire",
        "Online Order Confirmation",
        "Shipping & Tracking Update",
        "Digital Receipt & Invoice",
        "Subscription Renewal Alert",
        "Refund & Credit Note",
        "Service Quotation",
        "Electricity & Water Bill",
        "Internet & Mobile Bill",
        "Credit Card Monthly Statement",
        "Insurance Premium Notice",
        "Rent & Property Tax",
        "Loan & EMI Reminder",
        "Flight Booking & E-Ticket",
        "Hotel & Airbnb Reservation",
        "Car Rental Confirmation",
        "Visa & Immigration Update",
        "Public Transport & Rail Ticket",
        "Travel Insurance Policy",
        "Two-Factor Auth (2FA) Code",
        "Password Reset Request",
        "Suspicious Login Alert",
        "API Key & Dev Credential",
        "Server Downtime Notification",
        "Software License Key",
        "Course Enrollment Confirm",
        "University & Exam Update",
        "Webinar & Workshop Invite",
        "Certification & Badge Award",
        "Library & Book Return",
        "Learning Resource Digest",
        "Personal Family Email",
        "Social Media Friend Request",
    ]

    with open(file_path, "r") as f:
        text = f.read()[:2000]

    result = classifier(text, labels, multi_label=False)
    return f"tag: [{result['labels'][0]}]"


if __name__ == "__main__":
    print(classify(sys.argv[1]))
