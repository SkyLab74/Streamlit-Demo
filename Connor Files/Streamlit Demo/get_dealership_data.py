# NECESSARY IMPORTS #

# for importing data from redshift
# from redshift_connection import make_connection
# import sqlalchemy as sa
import pandas as pd

# using script to get connection
# s, engine = make_connection()
#
# # reading in the sql query
# with open('dealership_data_query.sql') as file:  # insert different path here to execute different query
#     escaped_sql = sa.text(file.read())
# engine.execute(escaped_sql)
#
#
# def get_dealership_data():
#     with engine.connect() as connection:
#         result = connection.execute(escaped_sql)
#
#     dealership_data = pd.DataFrame(result, columns=[col for col in result.keys()])
#
#     return dealership_data


def join_dealer_on_id(ids, dealer_data):
    '''
    helper function to add dealership level data from list of dealership ids
    :param ids: list of dealership ids
    :param dealer_data: dataframe pulled from redshift with info for all dealerships (dealer_id, name, state, zipcode, type, rsm_id, rsm_first_name
    :return: dataframe joined on dealership id list with more dealership info and calculated fields
    '''
    ids_series = pd.Series(ids, name='dealership_id')

    merged_df = pd.merge(left=ids_series,
                         right=dealer_data,
                         how='left',
                         left_on='dealership_id',
                         right_on='dealership_id')

    # merged_df.rename(columns={}, inplace=True)
    merged_df = merged_df.astype({'dealership_id': 'int32'})

    return merged_df


def get_rsm_name_list(dealer_df):
    '''
    get list of all rsm names to use for filter button
    :param dealer_df: redshift pull of dealer data that is used to join on to data
    :return: list of each rsm name
    '''

    # limit list to one entry per name
    names = list(set(dealer_df.rsm_first_name.astype(str)))

    # remove nulls is exist
    names = list(filter(None, names))
    names.sort()

    return names

def get_state_list(dealer_df):
    '''
    get list of all rsm names to use for filter button
    :param dealer_df: redshift pull of dealer data that is used to join on to data
    :return: list of each rsm name
    '''

    # limit list to one entry per name and sort
    states = list(set(dealer_df.state.astype(str)))

    # remove nulls is exist and sort
    states = list(filter(None, states))
    states.sort()

    return states