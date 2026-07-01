# Retail Database Schema

A small SQLite retail dataset. All monetary values are USD. Dates are `YYYY-MM-DD`.

There are four tables: `customers`, `products`, `orders`, `order_items`.

---

## Table: customers

One row per customer.

| Column        | Type    | Description                                  |
|---------------|---------|----------------------------------------------|
| customer_id   | INTEGER | Primary key.                                 |
| first_name    | TEXT    | Customer first name.                         |
| last_name     | TEXT    | Customer last name.                          |
| state         | TEXT    | US state code (e.g. `CA`, `TX`, `NY`).       |
| signup_date   | DATE    | Date the customer signed up.                 |

---

## Table: products

One row per product in the catalog.

| Column        | Type    | Description                                          |
|---------------|---------|------------------------------------------------------|
| product_id    | INTEGER | Primary key.                                         |
| product_name  | TEXT    | Display name of the product.                         |
| category      | TEXT    | One of: Electronics, Home, Clothing, Sports, Books.  |
| price         | REAL    | Catalog list price (USD).                            |

---

## Table: orders

One row per order. An order belongs to exactly one customer and has one or more order items.

| Column        | Type    | Description                                                    |
|---------------|---------|----------------------------------------------------------------|
| order_id      | INTEGER | Primary key.                                                   |
| customer_id   | INTEGER | Foreign key -> customers.customer_id.                          |
| order_date    | DATE    | Date the order was placed.                                     |
| total_amount  | REAL    | Order total (USD); equals the sum of its order_items lines.    |

---

## Table: order_items

Line items within an order. One row per (order, product).

| Column        | Type    | Description                                           |
|---------------|---------|-------------------------------------------------------|
| order_item_id | INTEGER | Primary key.                                          |
| order_id      | INTEGER | Foreign key -> orders.order_id.                       |
| product_id    | INTEGER | Foreign key -> products.product_id.                   |
| quantity      | INTEGER | Units of the product purchased in this line.          |
| unit_price    | REAL    | Price paid per unit (may differ slightly from list).  |

---

## Relationships

- `customers` 1 --- N `orders`            (customers.customer_id = orders.customer_id)
- `orders`    1 --- N `order_items`        (orders.order_id = order_items.order_id)
- `products`  1 --- N `order_items`        (products.product_id = order_items.product_id)

## Useful conventions

- **Revenue** = `order_items.quantity * order_items.unit_price` (line revenue), or `orders.total_amount` at the order level.
- **Order value** = `orders.total_amount`.
- **"Last month" / time windows** should filter on `orders.order_date`.
- Data spans **2023-01-01** to **2024-12-31**.
- A "customer with no orders" is a customer whose `customer_id` never appears in `orders`.
- A "product never purchased" is a product whose `product_id` never appears in `order_items`.
