import sqlite3
from langchain_community.utilities import SQLDatabase

import pandas as pd
from sqlalchemy import create_engine


# To rmove the tables from db call this function
def remove_tables():
    conn = sqlite3.connect("my_database.db")
    cursor = conn.cursor()

    # Execute SQL command to drop the table
    cursor.execute("DROP TABLE IF EXISTS forcast_demand_table")

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()
    db = SQLDatabase.from_uri("sqlite:///my_database.db")
    db = db.get_usable_table_names()
    return f"Table Removed!!! \n\n\n Available tables: {db}"


# Load Excel data into a DataFrame
"""df = pd.read_excel("stock.csv")

# Create a connection to the SQLite database
engine = create_engine("sqlite:///my_database.db")

# Write DataFrame to SQL table
df.to_sql("forcast_demand_table", con=engine, if_exists="replace", index=False)

print("Data transferred successfully!")"""

db = SQLDatabase.from_uri("sqlite:///my_database.db")

remove_tables()
print("Available tables:", db.get_usable_table_names())
