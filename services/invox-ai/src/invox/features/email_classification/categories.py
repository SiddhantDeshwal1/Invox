# services/invox-ai/src/invox/features/email_classification/categories.py

"""
Hierarchical Email Categories for Invox AI.
Used for Top-down Multi-step Hierarchical (TMH) zero-shot classification.
"""

CATEGORY_HIERARCHY = {
    "Work": [
        "Meetings",
        "Projects",
        "Team & Admin",
        "External",
        "Job Search",
        "Infrastructure",
        "Others",
    ],
    "Finance": [
        "Banking",
        "Investments",
        "Taxes",
        "Bills & Utilities",
        "Loans",
        "Insurance",
        "Others",
    ],
    "Purchases": [
        "Orders",
        "Shipping",
        "Receipts",
        "Subscriptions",
        "Refunds",
        "Food Delivery",
        "Others",
    ],
    "Travel": [
        "Flights",
        "Accommodations",
        "Transit",
        "Visas",
        "Itineraries",
        "Rewards",
        "Others",
    ],
    "Education": [
        "Courses",
        "Assignments",
        "Competitions",
        "Certifications",
        "Seminars",
        "Research",
        "Others",
    ],
    "Security": [
        "OTP & 2FA",
        "Passwords",
        "Alerts",
        "Privacy",
        "Backups",
        "Licenses",
        "Others",
    ],
    "Social": [
        "Personal",
        "Platforms",
        "Events",
        "Community",
        "Dating",
        "Greetings",
        "Others",
    ],
    "Promotions": [
        "Newsletters",
        "Sales",
        "Product Updates",
        "Surveys",
        "Giveaways",
        "Fundraising",
        "Others",
    ],
    "Spam": [
        "Phishing",
        "Scams",
        "Junk Mail",
        "Fake Invoices",
        "Malware",
        "Others",
    ],
    "Extra": [
        "Health & Medical",
        "Fitness",
        "Legal & Civic",
        "Real Estate",
        "Family & Pets",
        "Hobbies",
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
