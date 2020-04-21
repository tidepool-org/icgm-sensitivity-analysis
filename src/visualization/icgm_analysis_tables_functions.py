#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iCGM Analyis Tables Functions
=============================
:File: icgm_analysis_tables_functions.py
:Description: A set of functions used to calculate the table visualizations
              for the iCGM Analysis Tables notebook
:Version: 0.0.1
:Created: 2020-04-21
:Authors: Jason Meno (jam)
:Dependencies: None
:License: BSD-2-Clause
"""
# %% Libraries
import pandas as pd
# import numpy as np

# %% Table Functions


def get_table_1():
    """Table 1. Age Breakdown"""

    table_cols = ['Age (years old)', 'Count']
    table_1 = pd.DataFrame(columns=table_cols)
    table_1['Age (years old)'] = [
                                  '< 7',
                                  '[7, 13]',
                                  '[14, 24]',
                                  '[25, 49]',
                                  '≥ 50'
    ]

    return table_1


def get_table_2():
    """Table 2. Duration Breakdown"""

    table_cols = ['T1D Duration (years)', 'Count']
    table_2 = pd.DataFrame(columns=table_cols)
    table_2['T1D Duration (years)'] = ['< 7', '[1, 4]', '≥ 5']

    return table_2


def get_table_5A():
    """Table 5A. iCGM Special Controls Results Example"""

    table_cols = ['Criterion', 'iCGM Special Controls', 'Batch Sensor Results']
    table_5A = pd.DataFrame(columns=table_cols)
    criterion = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    sc_thresh = [0.85, 0.70, 0.80, 0.98, 0.99,
                 0.99, 0.87, 1.0, 1.0, 0.99, 0.99]
    table_5A['Criterion'] = criterion
    table_5A['iCGM Special Controls'] = sc_thresh

    return table_5A


def get_table_5B():
    """Table 5B. Batch iCGM Sensor Characteristics."""

    table_cols = ['iCGM Batch Sensor Characteristics', 'Batch Sensor Value']
    table_6B = pd.DataFrame(columns=table_cols)
    table_6B['iCGM Batch Sensor Characteristics'] = [
                                                     'Bias Factor Range',
                                                     'Bias Drift Range',
                                                     'Noise',
                                                     'Delay',
                                                     'MARD',
                                                     'MBE'
    ]

    return table_6B


def get_table_8():
    """Table 8. Example Summary Table"""

    table_cols = [
        'Level of Analysis',
        'N',
        'Avg LBGI8hr (+/- SD)',
        'Median LBGI8hr Risk Score (+/- IQR)',
        'Avg DKAI8hr (+/- SD)',
        'Median DKAI8hr Risk Score (+/- IQR)'
    ]
    table_8 = pd.DataFrame(columns=table_cols)
    table_8['Level of Analysis'] = [
                                    'All Analyses Combined',
                                    'tempBasal (a1)',
                                    'correctionBolus (a2)',
                                    'mealBolus (a3)',
                                    'b1', 'b2', 'b3',
                                    'b4', 'b5', 'b6',
                                    'b7', 'b8', 'b9'
    ]

    return table_8


def get_table_9(table_8):
    """Table 9. Example Summary Table at the All-Analysis-Combined Level

    This is just a subset of table 8
    """

    analysis_level = ['All Analyses Combined']

    selected_rows = table_8['Level of Analysis'].isin(analysis_level)
    table_9 = table_8.loc[selected_rows]

    return table_9


def get_table_10(table_8):
    """Table 10. Example Summary Table at the Analysis-Level

    This is just a subset of table 8
    """

    analysis_level = [
            'tempBasal (a1)',
            'correctionBolus (a2)',
            'mealBolus (a3)'
    ]

    selected_rows = table_8['Level of Analysis'].isin(analysis_level)
    table_10 = table_8.loc[selected_rows]

    return table_10


def get_table_11():
    """Table 11. Spearman Correlation Coefficient Table (rho, and p-value)."""

    table_cols = [
        'Parameters',
        'LBGI8hr (rho, p)',
        'LBGI8hr Risk Score (rho, p)',
        'DKAI8hr (rho, p)',
        'DKAI8hr Risk Score (rho, p)'
    ]

    table_11 = pd.DataFrame(columns=table_cols)

    table_11['Parameters'] = [
                              'Bias Factor',
                              'Bias Drift Params',
                              'Noise Coefficient',
                              'MARD',
                              'MBE'
    ]
    return table_11


def get_table_12(table_8):
    """Table 12. Example Summary Table at the True BG Test Condition Level

    This is just a subset of table 8
    """

    analysis_level = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
    selected_rows = table_8['Level of Analysis'].isin(analysis_level)
    table_12 = table_8.loc[selected_rows]
    table_12 = table_12.rename(
            columns={'Level of Analysis': 'Level of Analysis \n(True BG)'}
    )

    return table_12


def get_table_13():
    """Table 13. Example Summary Table at the iCGM Test Condition Level"""

    table_13 = pd.DataFrame()

    return table_13


def get_table_17():
    """Table 17. Overall Risk of Severe Hypoglycemia"""
    table_cols = [
        'Level of Analysis iCGM Conditions',
        '30min Median BG (mg/dL) & 15min Rate of Change (mg/dL/min)',
        'TBDDP CGM Freq (Prob)',
        'Probability of Occurrence Rating (P1)',
        'Median LBGI8hr Risk Score (+/- IQR)',
        'Overall Risk (Prob X Severity RS)'
    ]
    table_17 = pd.DataFrame(columns=table_cols)

    conditions = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
    table_17['Level of Analysis iCGM Conditions'] = conditions

    return table_17


def get_table_18():
    """Table 18. Overall Risk of DKA"""
    table_cols = [
        'Level of Analysis iCGM Conditions',
        '30min Median BG (mg/dL) & 15min Rate of Change (mg/dL/min)',
        'TBDDP CGM Freq (Prob)',
        'Probability of Occurrence Rating (P1)',
        'Median DKAI8hr Risk Score (+/- IQR)',
        'Overall Risk (Prob X Severity RS)'
    ]
    table_18 = pd.DataFrame(columns=table_cols)

    conditions = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
    table_18['Level of Analysis iCGM Conditions'] = conditions

    return table_18
