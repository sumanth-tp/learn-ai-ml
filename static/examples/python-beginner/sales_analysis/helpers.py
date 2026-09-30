"""Small calculations shared by report scripts."""


def calculate_total(quantity, unit_price):
    """Multiply scalars or aligned pandas columns."""
    return quantity * unit_price


def format_currency(amount):
    """Format the lab's USD amounts for display."""
    return f"${amount:,.2f}"
