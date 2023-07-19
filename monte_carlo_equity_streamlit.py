import multiprocessing
import streamlit         as st
import pandas            as pd
import numpy             as np
import quantstats        as qs
import matplotlib.pyplot as plt
from   datetime          import timedelta
from   backtesting       import Backtest, Strategy
from   multiprocessing   import Pool


def generate_equity():
    pass


def main():
    st.markdown("Monte Carlo simulation for Equity curve")

    col01, col02, col03 = st.columns(3)
    with col01:
        start_date = st.date_input('Start Date', min_value=None, max_value=None, key=None)
    with col02:
        period_years = st.number_input('Years', min_value=1, step=1, value=3)
    with col03:
        trades = st.number_input('Number of Trades', min_value=10, step=10, value=150)
        pass

    end_date = (start_date + timedelta(days=period_years*365)) if start_date else None
    st.text(f'End Date: {end_date.strftime("%Y-%m-%d") if end_date else ""}')

    if st.button('Generate Equity'):
        if not start_date:
            st.error('Error: Start date is mandatory.')
        elif start_date >= end_date:
            st.error('Error: Invalid date range. Start date should be earlier than end date.')
        elif not trades:
            st.error('Error: Number of trades is mandatory.')
        else:
            generate_equity()
    
    pass

if __name__ == '__main__':
    main()