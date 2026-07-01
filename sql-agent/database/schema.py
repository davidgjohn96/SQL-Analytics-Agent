"""SQLAlchemy schema for the sample retail database.

The schema is intentionally small and easy to read. It is the single source of
truth for table/column structure used by both the seed script and the agent's
SQL validation step.
"""

from __future__ import annotations

import os

from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

# retail.db lives next to this file.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail.db")
DB_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    state = Column(String, nullable=False)
    signup_date = Column(Date, nullable=False)

    orders = relationship("Order", back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True)
    product_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    price = Column(Float, nullable=False)

    order_items = relationship("OrderItem", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    order_date = Column(Date, nullable=False)
    total_amount = Column(Float, nullable=False)

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


def get_engine(echo: bool = False):
    """Return a SQLAlchemy engine bound to retail.db."""
    return create_engine(DB_URL, echo=echo)


def get_sessionmaker(echo: bool = False):
    return sessionmaker(bind=get_engine(echo=echo))
