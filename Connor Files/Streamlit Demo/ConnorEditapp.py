####################
# NECESSARY IMPORTS#
####################

# core app function mechanics
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# importing custom functions
from cohort_data_transformation import cohortData, get_month_string, add_change_calculation, get_breakdown_table
from get_cohort_data import prep_data  # , get_cohort_data
from get_dealership_data import join_dealer_on_id, get_rsm_name_list, get_state_list  # , get_dealership_data
# from get_pdf import PDF

# app building
import streamlit as st
from fpdf import FPDF
#from io import BytesIO
import base64
#from matplotlib.backends.backend_pdf import PdfPages


# getting dealership data
# dealership_df = get_dealership_data() # for live connection to redshfit
@st.cache(allow_output_mutation=True)
def get_dealership_csv():
    return pd.read_csv('dealership_data.csv')  # running off csv


dealership_df = get_dealership_csv()


# getting cohort data
# data_raw = get_cohort_data() # for live connection to redshfit
@st.cache(allow_output_mutation=True)
def get_cohort_csv():
    return pd.read_csv('cohort_data.csv')


data_raw = get_cohort_csv()  # running off csv
data_clean = prep_data(data_raw)

# establishing sample class
#               clean_data,     df,  min_val, start_date,     end_date,      num_months,  rsm, state
data = cohortData(data_clean, data_clean, 5, date(2020, 2, 1), date(2021, 2, 1), 12, 'All', 'All')


def get_interactive_tool_data(mode_selection):
    # pivot data
    data.df = data.pivot_on_min_val()
    data.df = data.drop_incomplete_month()

    # get dealership id lists
    unretained_ids = data.get_unretained_ids()
    retained_ids = data.get_retained_ids()
    new_ids = data.get_new_ids()
    start_ids = data.get_dealership_ids(data.start_date)
    end_ids = data.get_dealership_ids(data.end_date)

    # apply rsm and state filters - join dealer data, then filter for each cohort subsection
    unretained_df = join_dealer_on_id(unretained_ids, dealership_df)
    unretained_df = data.filter_by_rsm(unretained_df)
    unretained_df = data.filter_by_state(unretained_df)

    retained_df = join_dealer_on_id(retained_ids, dealership_df)
    retained_df = data.filter_by_rsm(retained_df)
    retained_df = data.filter_by_state(retained_df)

    new_df = join_dealer_on_id(new_ids, dealership_df)
    new_df = data.filter_by_rsm(new_df)
    new_df = data.filter_by_state(new_df)

    start_df = join_dealer_on_id(start_ids, dealership_df)
    start_df = data.filter_by_rsm(start_df)
    start_df = data.filter_by_state(start_df)

    end_df = join_dealer_on_id(end_ids, dealership_df)
    end_df = data.filter_by_rsm(end_df)
    end_df = data.filter_by_state(end_df)

    # getting dealer counts
    retained_count = len(retained_df)
    unretained_count = len(unretained_df)
    new_count = len(new_df)
    start_count = len(start_df)
    end_count = len(end_df)

    # getting retention rate
    if start_count > 0:
        retention_rate = round((retained_count / start_count) * 100, 2)
    else:
        retention_rate = 0

    # displaying summary stats
    st.subheader('Overview')
    col1, col2 = st.beta_columns(2)
    col1.write(f'**{get_month_string(data.start_date)} {data.start_date.year}** Dealership Count :')
    col2.write(f'**{start_count}**')
    col1.write(f'Dealership Attrition:')
    col2.write(f'**-{unretained_count}**')
    col1.write(f'Dealerships Retained:')
    col2.write(f'**{retained_count}**')
    col1.write(f'Retention Rate:')
    col2.write(f'**{retention_rate}%**')
    col1.write(f'New Dealerships:')
    col2.write(f'**+{new_count}**')
    col1.write(f'**{get_month_string(data.end_date)} {data.end_date.year}** Dealership Count:')
    col2.write(f'**{end_count}**')

    # customizing contents of dealership level data view options based on prior function mode selection
    if mode_selection in ['Year-on-Year', 'Custom Date Range']:
        drill_down_selection = st.selectbox('View Dealerships', ['Unretained', 'Retained', 'New'])
    elif mode_selection in ['Month-on-Month']:
        drill_down_selection = st.selectbox('View Dealerships', ['Unretained', 'Retained'])

    # dealership level data
    if drill_down_selection == 'Unretained':
        # get dealership level data
        display_df = data.join_starting_status(unretained_df, is_new=False)
        display_df = data.join_ending_status(display_df)
        display_df = add_change_calculation(display_df)

        # get breakdown table
        breakdown = get_breakdown_table(display_df)

        # display breakdown table and dealership data
        st.subheader('Attrition Breakdown')
        st.dataframe(breakdown)

        st.subheader('Unretained Dealers')
        st.markdown('_(expand to see all fields)_')

        st.dataframe(display_df.style.background_gradient(cmap='RdYlGn', subset='%_change', vmin=-150, vmax=150))

        # download dealership data
        download = data.get_table_download_link(display_df, drill_down_selection)
        st.markdown(download, unsafe_allow_html=True)


    elif drill_down_selection == 'Retained':
        # get dealership level data
        display_df = data.join_starting_status(retained_df, is_new=False)
        display_df = data.join_ending_status(display_df)
        display_df = add_change_calculation(display_df)
        display_df = display_df.astype({'dealership_id': 'int32'})

        # get breakdown table
        breakdown = get_breakdown_table(display_df)

        # display breakdown table and dealership data
        st.subheader('Retention Breakdown')
        st.dataframe(breakdown)
        st.subheader('Retained Dealers')
        st.markdown('_(expand to see all fields)_')
        st.dataframe(display_df.style.background_gradient(cmap='RdYlGn', subset='%_change', vmin=-150, vmax=150))

        # download dealership data
        download = data.get_table_download_link(display_df, drill_down_selection)
        st.markdown(download, unsafe_allow_html=True)

    elif drill_down_selection == 'New':
        display_df = data.join_starting_status(new_df, is_new=True)
        display_df = data.join_ending_status(display_df)
        display_df = add_change_calculation(display_df)
        display_df = display_df.astype({'dealership_id': 'int32'})

        # get breakdown table
        breakdown = get_breakdown_table(display_df)

        # display breakdown table and dealership data
        st.subheader('New Dealership Breakdown')
        st.dataframe(breakdown)
        st.subheader('New Dealerships')
        st.markdown('_(expand to see all fields)_')
        st.dataframe(display_df.style.background_gradient(cmap='RdYlGn', subset='%_change', vmin=-150, vmax=150))

        # download dealership data
        download = data.get_table_download_link(display_df, drill_down_selection)
        st.markdown(download, unsafe_allow_html=True)

        st.write(
            'Please note that "start" date is one month prior to target date for new dealerships, since no loans were funded at true start date.')


def interactive_tool():
    # 1. sidebar selections

    # mode selection
    mode_list = ['Year-on-Year', 'Month-on-Month', 'Custom Date Range']
    mode = st.sidebar.selectbox('Mode', mode_list)

    # min e-level selection
    data.min_val = st.sidebar.slider("Min. engagement level", 1, 50, 10)

    # rsm filter selection
    rsm_names = get_rsm_name_list(dealership_df)
    rsm_names.insert(0, 'All')
    data.rsm = st.sidebar.selectbox('RSM Filter', rsm_names)

    # state filter selection
    state_names = get_state_list(dealership_df)
    state_names.insert(0, 'All')
    data.state = st.sidebar.selectbox('State Filter', state_names)

    # date selection - setting default menu selection to last completed month/year
    if date.today().month == 1:
        month_index = 12 - 1  # 12 to push to prior year, -1 because 0 index
        year_index = date.today().year - 1 - 2020  # -1 to push to prior year, -2020 to 0 index
    else:
        month_index = date.today().month - 1 - 1  # -1 to 0 index, -1 again to set to last complete month
        year_index = date.today().year - 2020  # -2020 to 0 index

    # date selection - generating date select boxes
    year = st.sidebar.selectbox('Target Year', range(2020, date.today().year + 1), index=year_index)
    month = st.sidebar.selectbox('Target Month', range(1, 13), index=month_index)

    # date selection - error handling
    if date(year, month, 1) >= date(date.today().year, date.today().month, 1):
        if date.today().month == 1:
            st.write(
                f'''Please select a target date between Jan 2020 and the most recent complete prior month 
                (i.e. select {get_month_string(date(year, 12, 1))} {date.today().year} if date is 
                {get_month_string(date.today())} {date.today().day}, {date.today().year}
                ''')
            return
        else:
            st.write(
                f'''Please select a target date between Jan 2020 and the most recent complete prior month 
                (i.e. select {get_month_string(date(year, date.today().month - 1, 1))} {date.today().year} if date is 
                {get_month_string(date.today())} {date.today().day}, {date.today().year}
                ''')
            return

    # 2. generate data based on mode selection
    if mode == 'Year-on-Year':
        st.header('Year-on-Year Retention')
        data.start_date = date(year - 1, month, 1)
        data.end_date = date(year, month, 1)

        get_interactive_tool_data(mode)


    elif mode == 'Month-on-Month':
        st.header('Month-on-Month Retention')

        # get correct date
        if month == 1:
            data.start_date = date(year - 1, 12, 1)
        else:
            data.start_date = date(year, month - 1, 1)

        data.end_date = date(year, month, 1)

        get_interactive_tool_data(mode)

    # elif mode == 'Year-to-Date':
    #     st.header('Year-to-Date Retention')
    #
    #     data.start_date = date(year, 1, 1)
    #     data.end_date = date(year, month, 1)
    #
    #     get_interactive_tool_data(mode)

    elif mode == 'Custom Date Range':
        st.header('Custom Date Range Retention')
        start_year = st.sidebar.selectbox('Start Year', range(2020, date.today().year + 1), index=0)
        start_month = st.sidebar.selectbox('Start Month', range(1, 13), index=0)

        data.start_date = date(start_year, start_month, 1)
        data.end_date = date(year, month, 1)

        # validate date order is correct
        if data.end_date <= data.start_date:
            st.write('Target date must be after start date. Please select new dates.')
        else:
            get_interactive_tool_data(mode)  # passing mode selection on to next function to determine proper options


def get_heatmap():
    # toggle for min engagement level
    data.min_val = st.sidebar.slider("Min. engagement level", 1, 50, 10)

    # get max length for num_months slider
    end_date = date.today()
    start_date = date(2019, 1, 1)
    max_length = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # toggle for heatmap size
    data.num_months = st.sidebar.slider("Months included", 1, max_length,
                                        12)  # set max to 12 for now for formatting reasons

    # transform data
    data.df = data.pivot_on_min_val()
    data.df = data.get_retention_counts_df()
    data.df = data.drop_incomplete_month()
    data.df = data.limit_x_months()

    # get counts heatmap
    counts_fig = data.generate_heatmap(is_percent=False)

    # get percents data and heatmap
    data.df = data.get_retention_percents_df()
    percents_fig = data.generate_heatmap(is_percent=True)

    # show viz
    st.subheader('Retention Rates')
    st.pyplot(percents_fig)

    st.subheader('Retention Counts')
    st.pyplot(counts_fig)

    # pdf download button
    result = st.sidebar.button("Export PDF")
    if result:  # if clicked
        st.sidebar.write("Generating download link (this may take a moment)")

        class PDF(FPDF):
            def header(self):
                # Logo - this is an actual file in the repo, taken from the website
                self.image('Octane-logo-og.png', 10, 8, 33)
                # Arial bold 15
                self.set_font('Arial', 'B', 15)
                # Move to the right
                self.cell(50)
                # Title
                self.cell(110, 15, 'Cohort Analysis by Engagement Type', 0, 0, 'C')
                # Line break
                self.ln(20)

            # Page footer
            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                # Arial italic 8
                self.set_font('Arial', 'I', 8)
                # Page number
                self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

        # instantiate class as defined above
        pdf = PDF()
        pdf.alias_nb_pages()

        pdf.add_page()
        pdf.set_font(family='arial', style='B', size=12)
        pdf.cell(w=0, h=12, txt='Retention Rates', ln=1, align='L')
        pdf.set_font(family='arial', style='', size=12)
        pdf.cell(w=0, h=16, txt=f'Minimum engagement level: {data.min_val}', ln=1, align='L')
        pdf.image(name='heatmap_percents.png', type='PNG', w=175, h=175)

        pdf.add_page()
        pdf.set_font(family='arial', style='B', size=12)
        pdf.cell(w=0, h=12, txt='Retention Counts', ln=1, align='L')
        pdf.set_font(family='arial', style='', size=12)
        pdf.cell(w=0, h=16, txt=f'Minimum engagement level: {data.min_val}', ln=1, align='L')
        pdf.image(name='heatmap_counts.png', w=175, h=175)


        def create_download_link(val, filename):
            b64 = base64.b64encode(val)  # val looks like b'...'
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

        html = create_download_link(pdf.output(dest="S").encode("latin-1"), f"retention_heatmap_e{data.min_val}_{date.today()}")
        st.sidebar.markdown(html, unsafe_allow_html=True)


def main():
    '''
    initializes app, applows user to select page
    '''

    st.title('Cohort Analysis by Engagement Level')

    # main site nav
    activities = ['Comparison Tool', 'Heatmap', 'Documentation']
    choice = st.sidebar.selectbox('Primary Navigation', activities)

    if choice == 'Heatmap':
        get_heatmap()

    elif choice == 'Documentation':
        about()

    elif choice == 'Comparison Tool':
        interactive_tool()


def about():
    '''
    Helper function for the about page
    '''
    st.header('Documentation')
    st.markdown(
        '''
        **Overview**  
        
        This tool was created by the Sales & Marketing Analytics Team to help you measure cohort retention and 
        attrition by minimum engagement level in monthly increments.   
        
        For the purposes of this tool, attrition simply means that if a dealer that achieved a certain level of engagement
        at the start of a period then failed to do achieve that same level of engagement at the end of the period, they 
        will be considered to have attrited and be marked as "unretained". For example, so if an E10+ dealership attrited 
        YoY, that doesn’t mean they necessarily didn’t transact a year later, it just means that the same month the 
        following year they funded less than 10 loans.  
        
        This tool is composed of two components, a heatmap and an interactive cohort comparison tool.
        
        **Heatmap**
        
        The heatmap has two parameters that you can adjust: minimum engagement level and number of months included.
        * Min. engagement level limits the dealerships counted in the viz to those that reached the specified minimum
        number of funded loans at the beginning of the period. 
        * The other parameter allows you to limit the number of months
        you would like to view in the viz.  
        
        Cohorts are represented by horizontal lines on this heatmap. To track retention of an individual cohort in 
        this heatmap, find the row in the y axis (labeled "Cohort") and start at the leftmost cell and read left to right,
        month to month, to track how many dealerships in the original cohort transacted at the specified minimum engagement
        level over time, up to the last complete prior month.  
        
        Retention Rates tracks retention as a percentage of the 
        original cohort, whereas Retantion Counts displays the raw number of dealers each month retained by a given cohort.  
        
        **Interactive Cohort Comparison Tool**
        
        Like the heatmap, the interactive tool allows you to analyze cohort retention over a period of time, but unlike 
        the heatmap, this tool gives you access to a more granular view of the individual dealerships within each cohort.
        
        There are four main parameters you can adjust on this tool: mode, minimum engagement level, RSM Filter, and Target Date.
        * Mode allows you to choose the interval across which you would like to analyze cohort retention, Year-on-Year, 
        month-on-month, as well as custom selection of any two months. 
        * Min. engagement level limits the dealerships counted in the viz to those that reached the specified minimum
        number of funded loans at the beginning of the period. 
        * RSM Filter limits results to the dealerships assigned to a particular RSM. All data in the tool is filtered 
        accordingly is "All" is not selected (e.g. the total dealership count for each month will only reflect those
        dealerships for which the selected RSM is responsible.
        * Target Year and Month allow you to select the end date of the period. For example, if you choose a target year 
        of Jan. 2021 and select M-o-M for the mode, the starging month of your period will be Dec. 2020.  
        
        You can view individual dealership data three ways in this tool, by those that were achieved the minimum engagement 
        level at the beginning and end of the period ("Retained"), by those that did not maintain that level of engagement 
        at the end of the period ("Unretained"/"Attrition"), and by those that were not in the original cohort but managed 
        to achieved the min. engagement level at the end month ("New"). You may select which of these groups you would like 
        to drill down on with the toggle below the "Overview" section statistics.
        
        The breakdown table allows you to see an overview of what status level each dealer was in at the start and end of the period. 
        
        The final table displays granular, dealership-level data about dealerships in the cohort. Click the arrows at the 
        top right of the table to expand the table and see, in addition to basic info, where each dealership was at the 
        beginning and end of the period. 
        * start_count: funded loand at the beginning of the period
        * start_status: E-level status at the beginning of the period
        * end_count: funded loand at the end of the period
        * end_status: E-level status at the end of the period
        * change_count: difference between start_count and end_count
        * %_change: percentage difference in funded loans between start and end of period (change_count / start_count)
        
        You can click on column labels to sort by the values in each column. Finally, you can download the dealership-level 
        data using the hyperlinked button at the bottom of the tool. 
        '''
    )


# starts app
if __name__ == "__main__":
    main()

