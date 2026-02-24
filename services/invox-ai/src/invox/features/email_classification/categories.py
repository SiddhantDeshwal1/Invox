# services/invox-ai/src/invox/features/email_classification/categories.py

"""
Hierarchical Email Categories for Invox AI.
Used for Top-down Multi-step Hierarchical (TMH) zero-shot classification.
"""

CATEGORY_HIERARCHY = {
    "Work": [
        "Meetings",
        "Projects",
        "Payroll",
        "Clients",
        "Recruitment",
        "Schedules",
        "Memos",
        "Approvals",
        "Training",
        "Reviews",
        "Expenses",
        "Contracts",
        "Strategy",
        "Equipment",
        "Onboarding",
        "Operations",
        "Others",
    ],
    "Finance": [
        "Banking",
        "Investments",
        "Taxes",
        "Bills",
        "Loans",
        "Insurance",
        "Crypto",
        "Mortgages",
        "Grants",
        "Others",
    ],
    "Purchases": [
        "Orders",
        "Shipping",
        "Subscriptions",
        "Refunds",
        "Groceries",
        "Services",
        "Receipts",
        "Invoices",
        "Memberships",
        "Others",
    ],
    "Travel": [
        "Flights",
        "Hotels",
        "Transit",
        "Itineraries",
        "Visas",
        "Rewards",
        "Trains",
        "Cruises",
        "Rentals",
        "Others",
    ],
    "Education": [
        "Classes",
        "Assignments",
        "Notices",
        "Exams",
        "Certifications",
        "Grades",
        "Alumni",
        "Scholarships",
        "Admissions",
        "Others",
    ],
    "Security": [
        "Verification",
        "Passwords",
        "Alerts",
        "Privacy",
        "Fraud",
        "Devices",
        "Backups",
        "Breaches",
        "Compliance",
        "Others",
    ],
    "Social": [
        "Family",
        "Friends",
        "Platforms",
        "Events",
        "Groups",
        "Dating",
        "Networking",
        "Forums",
        "Messages",
        "Others",
    ],
    "Promotions": [
        "Newsletters",
        "Sales",
        "Updates",
        "Surveys",
        "Charity",
        "Giveaways",
        "Coupons",
        "Campaigns",
        "Partnerships",
        "Others",
    ],
    "Spam": [
        "Phishing",
        "Scams",
        "Junk",
        "Fakes",
        "Malware",
        "Clickbait",
        "Spoofing",
        "Blackmail",
        "Others",
    ],
    "Personal": [
        "Health",
        "Fitness",
        "Legal",
        "Housing",
        "Pets",
        "Hobbies",
        "Civic",
        "Medical",
        "Auto",
        "Others",
    ],
}


def get_parent_labels() -> list[str]:
    """Returns the list of 10 top-level categories for Step 1 classification."""
    return list(CATEGORY_HIERARCHY.keys())


def get_child_labels(parent_category: str) -> list[str]:
    """Returns the list of sub-categories for a specific parent for Step 2 classification."""
    return CATEGORY_HIERARCHY.get(parent_category, [])


def get_flat_labels() -> list[str]:
    """
    Returns the original flat list of categories (e.g., 'Work - Meetings').
    Useful for backwards compatibility with legacy benchmarking scripts.
    """
    flat_labels = []
    for parent, children in CATEGORY_HIERARCHY.items():
        for child in children:
            flat_labels.append(f"{parent} - {child}")
    return flat_labels
