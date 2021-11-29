# NECESSARY IMPORTS #

# for importing data from redshift
# from redshift_connection import make_connection
# import sqlalchemy as sa
import pandas as pd

# using script to get connection
# s, engine = make_connection()
#
# # reading in the sql query
# with open('cohort_data_query.sql') as file:  # insert different path here to execute different query
#     escaped_sql = sa.text(file.read())
# engine.execute(escaped_sql)
#
#
# def get_cohort_data():
#     with engine.connect() as connection:
#         result = connection.execute(escaped_sql)
#
#     cohort_data = pd.DataFrame(result, columns=[col for col in result.keys()])
#
#     return cohort_data


def prep_data(df):
    # making unified date column always set to first of the month
    # df['date'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(df.year_fund, df.month_fund)])
    df['date'] = pd.to_datetime(
        {'year': df.year_fund, 'month': df.month_fund, 'day': 1})  # changed from string method above

    # adding column to convert dealership id to numeric, skipping nulls (errors='coerce')
    df['dealership_id'] = pd.to_numeric(df['dealership_id'], errors='coerce')

    return df
