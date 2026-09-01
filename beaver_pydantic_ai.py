# Beaver's Choice Paper Company - a Pydantic AI Multi-Agent System
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
# loading the openai stuff
from openai import OpenAI
# loading the pydantic ai stuff
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# ------------------------------------------------------------------------------------------------------
# Utility functions to use in the multi-agent system (given by Udacity)
# ------------------------------------------------------------------------------------------------------

# Generate sample inventory
def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


# Initialize the database
def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


# Create a transaction
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


# Get all inventory
def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


# Get stock level
def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


# Get supplier delivery date
def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


# Get cash balance
def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


# Generate financial report
def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


# Search quote history
def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

# ------------------------------------------------------------------------------------------------------
# End of Udacity given Utility Functions
# ------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------
# Multi-Agent System Implementation
# ------------------------------------------------------------------------------------------------------

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Doh! OpenAI API key is not set, eh?")

# Set up OpenAI model
base_url = os.getenv("OPENAI_BASE_URL")
if not base_url:
    raise ValueError("Doh! OpenAI base URL is not set, eh?")

# OpenAIChatModel is from pydantic_ai.models.openai
model = OpenAIChatModel(
    "gpt-4o-mini",
    provider=OpenAIProvider( # OpenAIProvider is from pydantic_ai.providers.openai
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1",
    ),
)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# helpers from the starter code we need to wrap:
# inventory: get_all_inventory, get_stock_level, get_cash_balance,
#     get_supplier_delivery_date, generate_financial_report
# create_transaction -> stock_orders here, sales on the ordering agent
# quoting: search_quote_history


# ------------------------------------------------------------------------------------------------------
# Inventory Agent
# ------------------------------------------------------------------------------------------------------

# The agent definition (used below via @inventory_agent decorator)
inventory_agent = Agent(
    model,
    name="inventory_agent",
    deps_type=str, # deps is the request date in YYYY-MM-DD format
    instructions=(
        "You manage inventory for Beaver's Choice Paper Company. "
        "Use the item names from the catalog. "  # from README
        "Customers add extra words (like white, assorted, high-quality). Still use the catalog item of that type. "
        "If we don't sell that kind of item at all, say so. "
        "If they want more than we have, restock it. Don't ask first. "
        "Check stock, reorder when needed, and use the database. "  # from Project Overview
        "Always look at stock and cash before buying more. "
        "Do not spend more cash than we have. "
        "Report current stock, shortages, restock cost, and delivery date. "
        "Do not create sales. "
        "Never ever ask how to proceed. Just do it. "
    ),
)


# Helper function for catalog item
def catalog_item(item_name: str) -> dict | None:
    """
    Find a paper_supplies item by name (not case sensitive).
    Customers don't type the catalog exactly, so we also match a catalog name sitting inside what they wrote (substring matching). 
    """
    query = item_name.strip().lower() # strip whitespace and convert to lowercase
    if not query:
        return None

    for item in paper_supplies:
        if item["item_name"].strip().lower() == query: # we lowercased query so we need to lowercase item["item_name"] as well
            return item # ... but returning the original item not lowercased item_name

    # e.g. "heavy cardstock (white)" -> Cardstock, "A4 glossy paper" -> Glossy paper
    best = None
    best_len = 0
    best_end = -1
    for item in paper_supplies:
        name = item["item_name"].strip().lower()
        pos = query.find(name)
        if pos == -1:
            continue
        end = pos + len(name)
        # the name later in the phrase is usually the product (kraft paper envelopes -> Envelopes)
        if end > best_end or (end == best_end and len(name) > best_len):
            best = item
            best_len = len(name)
            best_end = end
    if best is not None:
        return best

    # a few names people use that aren't written like the catalog
    nicknames = [
        ("washi tape", "Decorative adhesive tape (washi tape)"),
        ("streamers", "Party streamers"),
        ("poster board", "Large poster paper (24x36 inches)"),
        ("poster boards", "Large poster paper (24x36 inches)"),
        ("posters", "Poster paper"),
        ("printer paper", "A4 paper"),
        ("printing paper", "A4 paper"),
        ("copy paper", "Standard copy paper"),
        ("napkins", "Paper napkins"),
    ]
    for nickname, real_name in nicknames:
        if nickname in query:
            for item in paper_supplies:
                if item["item_name"] == real_name:
                    return item
    return None

# The tools:

@inventory_agent.tool_plain() # plain means no context is passed to the tool
def list_catalog_items() -> List[str]: # returns a list of strings
    """ 
    List every catalog items name that the company sells.
    """
    catalog_items = []
    for item in paper_supplies:
        catalog_items.append(item["item_name"])
    return catalog_items


@inventory_agent.tool
def list_inventory(ctx: RunContext[str]) -> dict[str, int]:
    """
    All items with positive stock.
    """
    # get_all_inventory already returns {item_name: quantity}
    return get_all_inventory(ctx.deps)


@inventory_agent.tool
def check_stock(ctx: RunContext[str], item_name: str) -> dict:
    """
    Current stock for one item.
    """
    item = catalog_item(item_name)
    if item is None:
        return {"message": f"unknown item: {item_name}"}

    stock_df = get_stock_level(item["item_name"], ctx.deps) # ctx.deps = date in YYYY-MM-DD
    if stock_df.empty:
        current_stock = 0
    else:
        current_stock = int(stock_df["current_stock"].iloc[0] or 0)
    return {"item_name": item["item_name"], "current_stock": current_stock}


@inventory_agent.tool
def assess_stock(ctx: RunContext[str], item_name: str, requested_quantity: int) -> dict:
    """
    Assess quantity of requested item vs stock and when to restock.
    """
    stock_result = check_stock(ctx, item_name) # using the check_stock tool to get the current stock
    if stock_result.get("message"):
        return stock_result

    exact_name = stock_result["item_name"]
    stock = stock_result["current_stock"]
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    match = inventory_df[inventory_df["item_name"] == exact_name]
    # if the item is found in the inventory database, get the min stock level
    if not match.empty:
        min_stock = int(match.iloc[0]["min_stock_level"])
        below_reorder_point = stock < min_stock
        can_fulfill = requested_quantity <= stock
    else:
        min_stock = None
        below_reorder_point = False
        can_fulfill = False
    shortfall = max(requested_quantity - stock, 0)
    return {
        "item_name": exact_name,
        "requested_quantity": requested_quantity,
        "current_stock": stock,
        "min_stock_level": min_stock,
        "can_fulfill": can_fulfill,
        "shortfall": shortfall,
        "below_reorder_point": below_reorder_point,
    }


@inventory_agent.tool
def check_cash_balance(ctx: RunContext[str]) -> dict:
    """
    Cash on hand. Check this before restocking so we don't spend more than we have.
    """
    cash = float(get_cash_balance(ctx.deps))
    return {"as_of_date": ctx.deps, "cash_balance": cash}


@inventory_agent.tool
def estimate_supplier_delivery_date(ctx: RunContext[str], quantity: int) -> dict:
    """
    Estimate the supplier delivery date for restocking an item with a certain quantity
    for a specific order date.
    """
    delivery_date = get_supplier_delivery_date(ctx.deps, quantity) # does not need item_name as input because it is not used in the function
    return {
        "quantity": quantity,
        "order_date": ctx.deps,
        "delivery_date": delivery_date,
    }


@inventory_agent.tool
def create_stock_order(ctx: RunContext[str], item_name: str, quantity: int) -> dict:
    """
    Record a stock purchase. Price is qty times catalog unit_price.
    """
    item = catalog_item(item_name)
    if item is None:
        return {"message": f"unknown item: {item_name}"}
    if quantity <= 0:
        return {"message": "quantity must be positive"}

    unit_price = float(item["unit_price"])
    total_price = quantity * unit_price
    transaction_id = create_transaction(
        item["item_name"],
        "stock_orders",
        quantity,
        total_price,
        ctx.deps,
    )
    return {
        "item_name": item["item_name"],
        "quantity": quantity,
        "total_price": total_price,
        "transaction_id": transaction_id,
    }


@inventory_agent.tool
def restock_item(ctx: RunContext[str], item_name: str, requested_quantity: int = 0) -> dict:
    """
    Buy stock when they want more than we have, or we're below the reorder point.
    Checks cash first. If cash availble is more than cost -> write a stock_orders transaction
    """
    if requested_quantity < 0:
        return {"message": "quantity must be positive"}

    assessment = assess_stock(ctx, item_name, requested_quantity)
    if assessment.get("message"):
        return assessment

    item_name = assessment["item_name"]
    stock = assessment["current_stock"]
    shortfall = assessment["shortfall"] # requested_quantity - stock
    min_stock = assessment["min_stock_level"]

    # buy the shortfall, or enough to get back over min stock, whichever is bigger
    restock_quantity = shortfall
    if assessment["below_reorder_point"] and min_stock is not None:
        restock_quantity = max(restock_quantity, min_stock - stock)

    if restock_quantity <= 0:
        return {"message": "no restock needed", "item_name": item_name, "current_stock": stock}

    item = catalog_item(item_name) # only to get unit_price
    restock_cost = restock_quantity * float(item["unit_price"])
    cash = float(get_cash_balance(ctx.deps))

    if cash < restock_cost:
        return {"message": "not enough cash", "restock_cost": restock_cost, "cash_balance": cash}

    order = create_stock_order(ctx, item_name, restock_quantity)
    if order.get("message"):
        return order

    delivery = estimate_supplier_delivery_date(ctx, restock_quantity)
    return {
        "item_name": item_name,
        "quantity": restock_quantity,
        "restock_cost": restock_cost,
        "transaction_id": order["transaction_id"],
        "delivery_date": delivery["delivery_date"],
    }


@inventory_agent.tool
def get_financial_report(ctx: RunContext[str]) -> dict:
    """
    Cash, inventory value, and assets.
    """
    return generate_financial_report(ctx.deps)


# ------------------------------------------------------------------------------------------------------
# Quoting Agent
# ------------------------------------------------------------------------------------------------------

quoting_agent = Agent(
    model,
    name="quoting_agent",
    deps_type=str, # same request date as the other agents, even if we don't write to the db
    instructions=(
        "You make quotes. "
        "Use the item names from catalog. "
        "If they added extra words, still quote the catalog item of that type. "
        "If we don't sell that kind of thing, skip it. Don't swap in a random other product. "
        "Take a look at old quotes if they help. "
        "This is one order. Call build_quote once with every item we can sell. "
        "Don't quote items one at a time, the bulk discount is on the combined units. "
        "Always put a bulk discount on quote and say why. "
        "Use build_quote for the total. Don't invent a price. " # othewise discounts are not applied
        "When you give the quote, say the discount percent and why. "
    ),
)


@quoting_agent.tool_plain()
def lookup_quote_history(search_terms: List[str]) -> list:
    """
    Past quotes matching these keywords.
    Keep the list short, the search ANDs the terms together.
    """
    if not search_terms:
        return {"message": "need some search terms"}
    return search_quote_history(search_terms)


@quoting_agent.tool_plain()
def lookup_unit_price(item_name: str) -> dict:
    """
    Catalog unit price for one item.
    """
    item = catalog_item(item_name)
    if item is None:
        return {"message": f"unknown item: {item_name}"}
    return {"item_name": item["item_name"], "unit_price": float(item["unit_price"])}


@quoting_agent.tool_plain()
def build_quote(item_names: List[str], quantities: List[int]) -> dict:
    """
    Itemized quote with a bulk discount on the whole order.
    Pass every item and qty together. Don't call this once per item.
    """
    if len(item_names) != len(quantities):
        return {"message": "item names and quantities don't match"}
    if len(item_names) == 0:
        return {"message": "need at least one item"}
    
    # no errors -> build the quote
    line_items = []
    subtotal = 0.0
    total_units = 0
    for i in range(len(item_names)):
        item = catalog_item(item_names[i])
        if item is None:
            return {"message": f"unknown item: {item_names[i]}"}
        qty = quantities[i]
        if qty <= 0:
            return {"message": "quantity must be positive"}
        unit_price = float(item["unit_price"])
        line_total = qty * unit_price
        line_items.append({
            "item_name": item["item_name"],
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total,
        })
        subtotal += line_total
        total_units += qty

    # README says every quote should include a bulk discount and WHY (!)
    if total_units >= 1000:
        discount_rate = 0.15
        why = "15% bulk discount because the order is 1000 or more units" 
    elif total_units >= 500:
        discount_rate = 0.10
        why = "10% bulk discount because the order is 500 or more units"
    elif total_units >= 100:
        discount_rate = 0.05
        why = "5% bulk discount because the order is 100 or more units"
    else:
        discount_rate = 0.02
        why = "2% bulk discount on this order"

    discount_amount = subtotal * discount_rate
    total = subtotal - discount_amount
    return {
        "line_items": line_items,
        "subtotal": round(subtotal, 2),
        "discount_percent": int(discount_rate * 100),
        "discount_amount": round(discount_amount, 2),
        "total": round(total, 2),
        "why": why,
    }


# ------------------------------------------------------------------------------------------------------
# Ordering Agent
# ------------------------------------------------------------------------------------------------------

ordering_agent = Agent(
    model,
    name="ordering_agent",
    deps_type=str, # deps is the request date in YYYY-MM-DD format
    instructions=(
        "You close the sale if we actually have the paper. "
        "Don't buy stock and don't make up a new price. "
        "Say if it shipped and when it should arrive. "
    ),
)


@ordering_agent.tool
def get_stock(ctx: RunContext[str], item_name: str) -> dict:
    """
    Current stock for one item. Check this before selling.
    """
    return check_stock(ctx, item_name)


@ordering_agent.tool
def verify_fulfillment(ctx: RunContext[str], item_name: str, quantity: int) -> dict:
    """
    Can we ship this qty from stock.
    """
    stock_result = check_stock(ctx, item_name)
    if stock_result.get("message"):
        return stock_result
    stock = stock_result["current_stock"]
    return {
        "item_name": stock_result["item_name"],
        "quantity": quantity,
        "current_stock": stock,
        "can_ship": quantity <= stock,
    }


@ordering_agent.tool
def estimate_delivery_date(ctx: RunContext[str], quantity: int) -> dict:
    """
    Delivery date for this order size.
    """
    # same helper as restock, just how long the qty takes
    return estimate_supplier_delivery_date(ctx, quantity)


@ordering_agent.tool
def create_sale(ctx: RunContext[str], item_name: str, quantity: int, price: float) -> dict:
    """
    Record a sale. Price comes from the quote for this line.
    """
    item = catalog_item(item_name)
    if item is None:
        return {"message": f"unknown item: {item_name}"}
    if quantity <= 0:
        return {"message": "quantity must be positive"}
    if price < 0:
        return {"message": "price can't be negative"}

    transaction_id = create_transaction(
        item["item_name"],
        "sales",
        quantity,
        price,
        ctx.deps,
    )

    return {
        "item_name": item["item_name"],
        "quantity": quantity,
        "price": price,
        "transaction_id": transaction_id,
    }


@ordering_agent.tool
def place_sale(ctx: RunContext[str], item_name: str, quantity: int, price: float) -> dict:
    """
    Sell only if stock covers the qty.
    """
    check = verify_fulfillment(ctx, item_name, quantity)
    if check.get("message"):
        return check
    if not check["can_ship"]:
        return {
            "message": "not enough stock",
            "item_name": check["item_name"],
            "current_stock": check["current_stock"],
        }

    sale = create_sale(ctx, item_name, quantity, price)
    if sale.get("message"):
        return sale

    delivery = estimate_delivery_date(ctx, quantity)
    return {
        "item_name": sale["item_name"],
        "quantity": quantity,
        "price": price,
        "transaction_id": sale["transaction_id"],
        "delivery_date": delivery["delivery_date"],
    
    }


# ------------------------------------------------------------------------------------------------------
# Orchestrator Agent
# ------------------------------------------------------------------------------------------------------

orchestrator = Agent(
    model,
    name="orchestrator",
    deps_type=str, # request date YYYY-MM-DD, passed through to the other agents
    instructions=(
        "You're the one talking to the customer. "
        "Don't touch the database. Send work to inventory, quoting, and ordering. "
        "Stock and restock first, then a quote, then sell what we can ship. "
        "If we're short, restock in this turn, then sell. Don't ask them to confirm. "
        "If only some items can ship, sell those and say why the rest didn't. "
        "This is one request, one quote. Call request_quote once with every item we can sell. "
        "Don't quote or discount each line as its own order. "
        "Don't ask them to pick another item. "
        "Customers won't use exact catalog names. Use the catalog item of that type "
        "(cardstock, colored paper, glossy paper, poster paper, A4 paper, washi tape, streamers) "
        "and say which catalog name you used. "
        "Skip a line only if we don't sell that kind of thing (balloons, tickets, cardboard). "
        "Still quote and sell the items we do have. "
        "Tell them the price, the bulk discount, if we can ship, and when it arrives. "
        "Use the quote incl. discounts from quoting, don't make up a price. " # otherwise it sometimes calculates the price itself without the bulk discount
        "If we can't do it, say why, don't print out internal errors. "
        "It is important that you mention the bulk discount percentage and why for the order as a whole. "

    ),
)


async def ask_agent(agent, question: str, date: str) -> str:
    result = await agent.run(question, deps=date)
    return result.output


@orchestrator.tool
async def consult_inventory(ctx: RunContext[str], question: str) -> str:
    """
    Ask inventory about stock and restocking.
    """
    return await ask_agent(inventory_agent, question, ctx.deps)


@orchestrator.tool
async def request_quote(ctx: RunContext[str], question: str) -> str:
    """
    Ask quoting for one priced offer covering every item we can sell.
    Send all items in this one question so the bulk discount is on the whole order.
    """
    return await ask_agent(quoting_agent, question, ctx.deps)


@orchestrator.tool
async def place_order(ctx: RunContext[str], question: str) -> str:
    """
    Ask ordering to record a sale if we can ship.
    """
    return await ask_agent(ordering_agent, question, ctx.deps)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine, seed=42) # the answer is 42
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # orchestrator + workers are defined above

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            result = orchestrator.run_sync(request_with_date, deps=request_date)
            response = result.output
        except Exception as e:
            print(f"Doh, request failed: {e}")
            response = "Couldn't do that one, eh?"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
