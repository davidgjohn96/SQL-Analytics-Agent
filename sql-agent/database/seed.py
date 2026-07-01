"""Generate deterministic fake retail data and write it to retail.db.

Run with:  python -m database.seed   (from the sql-agent/ directory)

All randomness is seeded so the database is byte-for-byte reproducible.
Approximate volumes: 500 customers, 100 products, 5,000 orders, 15,000 items.
"""

from __future__ import annotations

import datetime as dt
import random

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.schema import (
    Base,
    Customer,
    Order,
    OrderItem,
    Product,
    get_engine,
)

SEED = 42

N_CUSTOMERS = 500
N_PRODUCTS = 100
N_ORDERS = 5_000
AVG_ITEMS_PER_ORDER = 3.4  # -> ~15,000 order_items

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Maria",
    "Daniel", "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark",
    "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Andrew", "Emily",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
]
STATES = [
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]

# (category, [product base names], price range)
CATEGORIES = {
    "Electronics": (["Headphones", "Laptop", "Monitor", "Keyboard", "Webcam",
                     "Speaker", "Charger", "Tablet", "Smartwatch", "Router"],
                    (25, 1500)),
    "Home": (["Lamp", "Blender", "Cookware Set", "Vacuum", "Pillow",
              "Towel Set", "Knife Set", "Coffee Maker", "Air Fryer", "Kettle"],
             (15, 400)),
    "Clothing": (["T-Shirt", "Jeans", "Jacket", "Sneakers", "Hoodie",
                  "Socks", "Hat", "Scarf", "Belt", "Dress"],
                 (10, 200)),
    "Sports": (["Yoga Mat", "Dumbbells", "Running Shoes", "Water Bottle",
                "Bicycle", "Tennis Racket", "Backpack", "Jump Rope",
                "Resistance Bands", "Helmet"],
               (8, 800)),
    "Books": (["Novel", "Cookbook", "Biography", "Textbook", "Comic",
               "Journal", "Atlas", "Dictionary", "Guidebook", "Anthology"],
              (5, 60)),
}

START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date(2024, 12, 31)


def _random_date(rng: random.Random, start: dt.date, end: dt.date) -> dt.date:
    delta = (end - start).days
    return start + dt.timedelta(days=rng.randint(0, delta))


def build_customers(rng: random.Random) -> list[Customer]:
    customers = []
    for i in range(1, N_CUSTOMERS + 1):
        customers.append(
            Customer(
                customer_id=i,
                first_name=rng.choice(FIRST_NAMES),
                last_name=rng.choice(LAST_NAMES),
                state=rng.choice(STATES),
                signup_date=_random_date(rng, START_DATE, END_DATE),
            )
        )
    return customers


def build_products(rng: random.Random) -> list[Product]:
    products = []
    pid = 1
    # Spread products across categories until we hit N_PRODUCTS.
    cat_names = list(CATEGORIES.keys())
    while pid <= N_PRODUCTS:
        category = cat_names[(pid - 1) % len(cat_names)]
        bases, (lo, hi) = CATEGORIES[category]
        base = bases[(pid - 1) % len(bases)]
        # Append an index so names stay unique-ish (e.g. "Laptop Pro 3").
        suffix = rng.choice(["Pro", "Plus", "Lite", "Max", "Eco", "Classic"])
        price = round(rng.uniform(lo, hi), 2)
        products.append(
            Product(
                product_id=pid,
                product_name=f"{base} {suffix} {pid}",
                category=category,
                price=price,
            )
        )
        pid += 1
    return products


def build_orders_and_items(
    rng: random.Random, products: list[Product]
) -> tuple[list[Order], list[OrderItem]]:
    orders: list[Order] = []
    items: list[OrderItem] = []
    item_id = 1

    for oid in range(1, N_ORDERS + 1):
        customer_id = rng.randint(1, N_CUSTOMERS)
        order_date = _random_date(rng, START_DATE, END_DATE)

        n_items = max(1, int(rng.gauss(AVG_ITEMS_PER_ORDER, 1.2)))
        chosen = rng.sample(products, k=min(n_items, len(products)))

        total = 0.0
        for product in chosen:
            quantity = rng.randint(1, 5)
            # Unit price wobbles slightly around list price (promos etc.).
            unit_price = round(product.price * rng.uniform(0.9, 1.0), 2)
            total += unit_price * quantity
            items.append(
                OrderItem(
                    order_item_id=item_id,
                    order_id=oid,
                    product_id=product.product_id,
                    quantity=quantity,
                    unit_price=unit_price,
                )
            )
            item_id += 1

        orders.append(
            Order(
                order_id=oid,
                customer_id=customer_id,
                order_date=order_date,
                total_amount=round(total, 2),
            )
        )

    return orders, items


def seed() -> None:
    rng = random.Random(SEED)
    engine = get_engine()

    # Fresh start every run -> deterministic output.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    customers = build_customers(rng)
    products = build_products(rng)
    orders, items = build_orders_and_items(rng, products)

    with Session(engine) as session:
        session.add_all(customers)
        session.add_all(products)
        session.add_all(orders)
        session.add_all(items)
        session.commit()

        counts = {
            "customers": session.query(func.count(Customer.customer_id)).scalar(),
            "products": session.query(func.count(Product.product_id)).scalar(),
            "orders": session.query(func.count(Order.order_id)).scalar(),
            "order_items": session.query(func.count(OrderItem.order_item_id)).scalar(),
        }

    print("Seeded retail.db:")
    for table, count in counts.items():
        print(f"  {table:<12} {count:>6,}")


if __name__ == "__main__":
    seed()
