import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import seaborn as sns
import base64
import numpy as np


class cohortData:
    def __init__(self, clean_data, df, min_val, start_date, end_date, num_months, rsm, state):
        self.clean_data = clean_data
        self.df = df
        self.min_val = min_val
        self.start_date = start_date
        self.end_date = end_date
        self.num_months = num_months
        self.rsm = rsm
        self.state = state

    def pivot_on_min_val(self):

        '''
        takes df after it has been pulled from redshift and prepped
        returns data filtered to specified min level and pivoted
        '''
        # filter for only dealers who have hit a month at the e-level specified by min_val
        filtered_df = self.df[self.df.counts >= self.min_val]

        # pivot df to index by dealership_id
        pivot = filtered_df.pivot(index='dealership_id', columns='date', values='counts').fillna(0)

        # replace values with binary (did or did not achieve a status), since only values above min_val are displaying vals above min_val
        pivot[pivot > 0] = 1

        return pivot

    def get_dealership_ids(self, date):
        '''
        takes df and a date, returns list of dealership_ids that funded loans that month
        '''

        id_list = list(self.df[self.df[str(date)] > 0].index)
        id_list = [id for id in id_list if id > 0]  # >0 for null handling

        return id_list

    def get_retained_ids(self):
        '''
        determine which dealers were retained between months, returns list of dealers that were retained
        '''

        starting_dealers = list(self.df[self.df[str(self.start_date)] > 0].index)
        ending_dealers = list(self.df[self.df[str(self.end_date)] > 0].index)

        retained_ids = [id for id in starting_dealers if id in ending_dealers and id > 0]  # > 0 filters nans

        return retained_ids

    def get_unretained_ids(self):
        '''
        determine which dealers were retained between months, returns list of dealers that were retained
        '''
        starting_dealers = list(self.df[self.df[str(self.start_date)] > 0].index)
        ending_dealers = list(self.df[self.df[str(self.end_date)] > 0].index)

        retained_ids = [id for id in starting_dealers if id not in ending_dealers and id > 0]  # > 0 filters nans

        return retained_ids

    def get_new_ids(self):
        '''
        determine which dealers were retained between months, returns list of dealers that were retained
        '''
        starting_dealers = list(self.df[self.df[str(self.start_date)] > 0].index)
        ending_dealers = list(self.df[self.df[str(self.end_date)] > 0].index)

        retained_ids = [id for id in ending_dealers if id not in starting_dealers and id > 0]  # > 0 filters nans

        return retained_ids

    # def get_retained_id_count(self):
    #     '''
    #     determine which dealers were retained between months, returns list of dealers that were retained
    #     '''
    #     starting_dealers = get_dealer_ids(self.df, self.start_date)
    #     ending_dealers = get_dealer_ids(self.df, self.end_date)
    #
    #     retained_ids = [id for id in starting_dealers if id in ending_dealers]
    #
    #     if self.start_date > self.end_date:
    #         return 0
    #     else:
    #         return len(retained_ids)

    def get_retention_counts_df(self):
        # INNER FUNCTION
        def get_retained_id_count(df, date1, date2):
            '''
            determine which dealers were retained between months, returns list of dealers that were retained
            '''
            starting_dealers = list(df[df[date1] > 0].index)
            ending_dealers = list(df[df[date2] > 0].index)

            retained_ids = [id for id in starting_dealers if id in ending_dealers and id > 0]  # > 0 filters nans

            if date1 > date2:
                return 0
            else:
                return len(retained_ids)

        # MAIN FUNCTION
        # generate cohort analysis matrix like we have in google sheets
        new_df_dict = {}

        # iterating through each date once for each date (nested loop)
        # each key is a date, each value is a list of retained id counts in order of date for each month

        for i in self.df.columns:
            new_df_dict[i] = []
            for j in self.df.columns:
                new_df_dict[i].append(get_retained_id_count(self.df, i, j))

        df_final_counts = pd.DataFrame.from_dict(new_df_dict, orient='index', columns=new_df_dict.keys())
        df_final_counts[df_final_counts == 0] = np.NaN

        return df_final_counts

    def get_retention_percents_df(self):
        '''
        takes dataframe with rows and cols as months, values as counts
        returns same df but with values as percentages based off of first count in cohort
        '''
        cohort_max = self.df.max()
        retention = self.df.divide(cohort_max, axis=0)
        # retention.round(2) * 100
        return retention

    def limit_x_months(self):
        '''
        limits view of df to only last x months (spec. by num_months)
        '''
        df2 = self.df[self.df.columns[-(self.num_months):]]
        df3 = df2.iloc[-(self.num_months):]

        return df3

    def drop_incomplete_month(self):
        '''
        automatically removes column with current month until month is over
        '''
        if self.df.columns[-1].month == date.today().month:
            df2 = self.df[self.df.columns[:-1]]
            df3 = df2.iloc[:-1]

            return df3

        else:
            return self.df

    def join_starting_status(self, joined_df, is_new):
        # for retained and unretained dealer dfs
        if is_new == False:
            # filtering clean data to only include rows for start date
            original_data_filtered_1 = self.clean_data[self.clean_data.year_fund == self.start_date.year]
            original_data_filtered_2 = original_data_filtered_1[
                original_data_filtered_1.month_fund == self.start_date.month]

        # for new dealer df, since no data at start by definition
        elif is_new == True:
            # get correct date
            if self.end_date.month == 1:
                month_prior = date(self.end_date.year - 1, 12, 1)
            else:
                month_prior = date(self.end_date.year, self.end_date.month - 1, 1)

            # filtering clean data to only include rows for start date
            original_data_filtered_1 = self.clean_data[self.clean_data.year_fund == month_prior.year]
            original_data_filtered_2 = original_data_filtered_1[
                original_data_filtered_1.month_fund == month_prior.month]

        # joining dfs
        merged_df = pd.merge(left=joined_df,
                             right=original_data_filtered_2,
                             how='left',
                             left_on='dealership_id',
                             right_on='dealership_id')

        # dropping unnecessary/redundant cols
        merged_df.drop(columns=['date', 'month_fund', 'year_fund'], inplace=True)

        # renaming for clarity
        merged_df.rename(columns={'counts': 'start_count', 'engagement': 'start_status'}, inplace=True)

        # since is_new=True can result in nulls
        merged_df.start_count.fillna(0, inplace=True)
        merged_df.start_status.fillna('E0', inplace=True)

        merged_df = merged_df.astype({'start_count': 'int32'})

        return merged_df

    def join_ending_status(self, joined_df):
        # filtering clean data to only include rows for end date
        original_data_filtered_1 = self.clean_data[self.clean_data.year_fund == self.end_date.year]
        original_data_filtered_2 = original_data_filtered_1[original_data_filtered_1.month_fund == self.end_date.month]

        # joining dfs
        merged_df = pd.merge(left=joined_df,
                             right=original_data_filtered_2,
                             how='left',
                             left_on='dealership_id',
                             right_on='dealership_id')

        # dropping unnecessary/redundant cols
        merged_df.drop(columns=['date', 'month_fund', 'year_fund', 'zipcode', 'type', 'rsm_id', ], inplace=True)

        # renaming for clarity
        merged_df.rename(columns={'counts': 'end_count', 'engagement': 'end_status'}, inplace=True)

        # filling nulls (0s all come up null)
        merged_df.end_count.fillna(0, inplace=True)
        merged_df.end_status.fillna('E0', inplace=True)

        merged_df = merged_df.astype({'end_count': 'int32'})

        return merged_df

    def filter_by_rsm(self, joined_df):
        if self.rsm != 'All':
            joined_df = joined_df[joined_df.rsm_first_name == self.rsm]
        else:
            pass

        return joined_df

    def filter_by_state(self, joined_df):
        if self.state != 'All':
            joined_df = joined_df[joined_df.state == self.state]
        else:
            pass

        return joined_df

    def get_table_download_link(self, df, type_selection):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        file_name = str(self.start_date) + '_' + str(self.end_date) + '_' + type_selection.lower() + '_dealership_data'

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download CSV File</a>'

        return href

    def generate_heatmap(self, is_percent=False):
        rcParams.update({'figure.autolayout': False})

        # setting label text size according to number of months included, since plot changes size
        fontsize = 24  # need to start with initial size for heatmap
        if self.num_months > 26:
            fontsize = 14
        elif 22 <= self.num_months < 26:
            fontsize = 16
        elif 17 <= self.num_months < 22:
            fontsize = 20
        elif 14 <= self.num_months < 17:
            fontsize = 24
        elif 6 <= self.num_months < 14:
            fontsize = 30
        elif self.num_months < 6:
            fontsize = 36

        # generate heatmap
        fig, ax = plt.subplots(1, 1, figsize=(24, 24))
        # fig.suptitle('Retention Rates', fontsize=40)
        ticklabels = [str(col.month) + '/' + str(col.year - 2000) for col in self.df.columns]  # truncating tick labels

        # params based on counts or percents
        g = sns.heatmap(data=self.df,
                        annot=True,
                        cmap='RdYlGn' if is_percent is True else 'Greys',
                        # for transparent white ListedColormap(['grey'])
                        center=.5 if is_percent is True else None,
                        vmax=1 if is_percent is True else None,
                        vmin=0 if is_percent is True else None,
                        fmt='.0%' if is_percent is True else '.0f',
                        yticklabels=ticklabels,
                        xticklabels=ticklabels,
                        cbar_kws=dict(use_gridspec=False, location="bottom"),
                        annot_kws={'fontsize': fontsize, 'fontweight': 'bold'})

        # axis labels
        g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=fontsize)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=fontsize)

        # formatting color bar
        cbar = ax.collections[0].colorbar  # use matplotlib.colorbar.Colorbar object
        cbar.ax.tick_params(labelsize=fontsize)  # if is_percent is True else 0)  # here set the labelsize by 24

        # labeling and repositioning ticks
        ax.set_xlabel('\nMonth\n', fontsize=fontsize)
        ax.set_ylabel('\n            Cohort', fontsize=fontsize, rotation=0)
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('right')
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()

        if is_percent == True:
            plt.savefig('heatmap_percents.png', dpi=72, bbox_inches="tight", pad_inches=1)
        else:
            plt.savefig('heatmap_counts.png', dpi=72, bbox_inches="tight", pad_inches=1)
        return fig


######################################
########## helper functions ##########
######################################

def get_month_string(date):
    month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                  7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    return month_dict[date.month]


def add_change_calculation(df):
    # calculating value
    df['change_count'] = df.end_count - df.start_count
    df['%_change'] = (df.change_count / df.start_count)

    # sorting by change
    df.sort_values(by='change_count', ascending=True, inplace=True)
    df = df.astype({'change_count': 'int32'})

    # formatting % change
    df['%_change'] = df['%_change'].apply(lambda x: '{0:.2f}'.format(x * 100))

    return df


def get_breakdown_table(df):
    '''
    get start/end breakdown for status level of retained and unretained dealers in a given cohort
    :param df: df that is ready to display with dealer-level breakdown, with start and end status joined
    :return: df with count and % of dealers in each status at start and end of period
    '''
    # get status counts for start/end and merge
    start_breakdown = df.start_status.value_counts()
    end_breakdown = df.end_status.value_counts()
    breakdown = pd.merge(left=start_breakdown, right=end_breakdown, how='outer', left_index=True, right_index=True)

    # formatting
    breakdown = breakdown.reindex(['E0', 'E1', 'E2', 'E5', 'E10'])
    breakdown = breakdown.rename({'E2':'E2-4', 'E5':'E5-9', 'E10':'E10+'}, axis=0)
    breakdown.fillna(0, inplace=True)
    breakdown.rename(columns={'start_status': 'start_count', 'end_status': 'end_count'}, inplace=True)

    # add %s cols, round and display as x.xx%
    breakdown['start_%'] = breakdown.start_count / sum(breakdown.start_count)
    breakdown['start_%'] = breakdown['start_%'].apply(lambda x: '{0:.2f}%'.format(x * 100))
    breakdown['end_%'] = breakdown.end_count / sum(breakdown.end_count)
    breakdown['end_%'] = breakdown['end_%'].apply(lambda x: '{0:.2f}%'.format(x * 100))

    breakdown = breakdown[['start_count', 'start_%', 'end_count', 'end_%']]

    breakdown = breakdown.round(2)
    return breakdown
