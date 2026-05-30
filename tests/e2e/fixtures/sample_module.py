"""
Fixture Python module used by the M11 E2E ingestion test.

This file is intentionally simple so tree-sitter can parse it deterministically
and graph_builder can index its call edges.
"""


def calculate_discount(price: float, pct: float) -> float:
    """Apply a percentage discount to a price."""
    if pct < 0 or pct > 100:
        raise ValueError(f"Invalid discount percentage: {pct}")
    return price * (1 - pct / 100)


def apply_tax(price: float, rate: float) -> float:
    """Apply a tax rate to a price."""
    return price * (1 + rate)


def process_order(items: list, discount: float = 0, tax_rate: float = 0.1) -> float:
    """Process an order: sum items, apply discount, then add tax."""
    subtotal = sum(item["price"] * item["qty"] for item in items)
    discounted = calculate_discount(subtotal, discount)
    return apply_tax(discounted, tax_rate)


class PricingEngine:
    """Simple pricing engine that wraps the free functions."""

    def __init__(self, default_tax: float = 0.1):
        self.default_tax = default_tax

    def quote(self, items: list, discount: float = 0) -> float:
        return process_order(items, discount=discount, tax_rate=self.default_tax)
