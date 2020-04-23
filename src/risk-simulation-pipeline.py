#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Simulation Pipeline
========================
:File: risk-simulation-pipeline.py
:Description: A full simulation pipeline to prepare data for Tidepool's
iCGM Sensitivity Analysis Plan
:Version: 0.0.1
:Created: 2020-03-02
:Authors: Jason Meno (jameno)
:Dependencies:
    * icgm_simulator.py - Generates iCGM sensors and iCGM traces
    * simulator_functions.py - Helper functions used in the icgm simulator
    * pyloopkit_risk_simulator.py - Simulates Loop and diabetes metabolism
    * A folder containing PyLoopKit condition scenario .csv files
:License: BSD-2-Clause
"""

# %% Import Libraries

import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
import traceback
import sys
import datetime

import icgm_simulator as icgm_sim
import simulator_functions as sf
import pyloopkit_risk_simulator as loop_sim

import warnings
warnings.filterwarnings("ignore")  # Filtering user warnings

# %% Main Function Pipeline


def create_results_template():
    # Create empty template for appending simulation results
    sim_results_columns = [
        'file_name',
        'analysis_type',
        'virtual_patient_num',
        'bg_test_condition',
        'icgm_sensor_num',
        'icgm_test_condition',
        'age',
        'ylw',
        'CIR',
        'ISF',
        'BR',
        'SEG_RS',
        'pump_LBGI',
        'pump_LBGI_RS',
        'pump_HBGI',
        'pump_HBGI_RS',
        'loop_LBGI',
        'loop_LBGI_RS',
        'loop_HBGI',
        'loop_HBGI_RS',
        'DKAI',
        'DKAI_RS',
        'sensor_initial_bias',
        'sensor_phi_drift',
        'sensor_bias_drift_range_start',
        'sensor_bias_drift_range_end',
        'sensor_bias_drift_oscillations',
        'sensor_bias_norm_factor',
        'sensor_noise_coefficient',
        'sensor_delay',
        'sensor_random_seed',
        'sensor_bias_factor',
        'sensor_noise_seed',
        'sensor_noise',
        'sensor_drift_multiplier',
        'sensor_MARD',
        'sensor_MBE',
        'sensor_MBE_UB95',
        'sensor_CV',
        'sensor_SNR',
        'sensor_ICGM_PASS%',
        'sensor_LOSS_SCORE',
        'sensor_A_nPairs',
        'sensor_B_nPairs',
        'sensor_C_nPairs',
        'sensor_D_nPairs',
        'sensor_E_nPairs',
        'sensor_F_nPairs',
        'sensor_G_nPairs',
        'sensor_H_nPairs',
        'sensor_I_nPairs',
        'sensor_J_nPairs',
        'sensor_K_nPairs',
        'sensor_A_results',
        'sensor_B_results',
        'sensor_C_results',
        'sensor_D_results',
        'sensor_E_results',
        'sensor_F_results',
        'sensor_G_results',
        'sensor_H_results',
        'sensor_I_results',
        'sensor_J_results',
        'sensor_K_results',
        'n_sensors',
        'use_g6_accuracy_in_loss',
        'bias_type',
        'bias_drift_type',
        'delay',
        'random_seed',
        'TRUE.kind',
        'TRUE.N',
        'TRUE.min_value',
        'TRUE.max_value',
        'TRUE.time_interval',
        'SPECIAL_CONTROLS_CRITERIA',
        'SEARCH_SPAN',
        'BIAS_CATEGORY',
        'BIAS_MIN',
        'BIAS_MAX',
        'BIAS_DRIFT_MIN',
        'BIAS_DRIFT_MAX',
        'BIAS_DRIFT_STEP',
        'BIAS_DRIFT_OSCILLATION_MIN',
        'BIAS_DRIFT_OSCILLATION_MAX',
        'BIAS_DRIFT_OSCILLATION_STEP',
        'NOISE_MIN',
        'NOISE_MAX',
        'NOISE_STEP',
        'a',
        'b',
        'mu',
        'sigma',
        'batch_noise_coefficient',
        'bias_drift_range_min',
        'bias_drift_range_max',
        'batch_bias_drift_oscillations',
        'MARD',
        'MBE',
        'MBE_UB95',
        'CV',
        'SNR',
        'ICGM_PASS%',
        'LOSS_SCORE',
        'A_nPairs',
        'B_nPairs',
        'C_nPairs',
        'D_nPairs',
        'E_nPairs',
        'F_nPairs',
        'G_nPairs',
        'H_nPairs',
        'I_nPairs',
        'J_nPairs',
        'K_nPairs',
        'A_results',
        'B_results',
        'C_results',
        'D_results',
        'E_results',
        'F_results',
        'G_results',
        'H_results',
        'I_results',
        'J_results',
        'K_results',
        '95%LB_percentWithin20/20%YSI',
        'MARD%',
        "('[40, 54)', 'percentWithin15YSI')",
        "('[40, 54)', 'percentWithin20YSI')",
        "('[40, 54)', 'percentWithin40YSI')",
        "('[40, 54)', 'MBE')",
        "('[40, 54)', 'MARD%')",
        "('[54, 70)', 'percentWithin15YSI')",
        "('[54, 70)', 'percentWithin20YSI')",
        "('[54, 70)', 'percentWithin40YSI')",
        "('[54, 70)', 'MBE')",
        "('[54, 70)', 'MARD%')",
        "('[70, 180]', 'percentWithin15%YSI')",
        "('[70, 180]', 'percentWithin20%YSI')",
        "('[70, 180]', 'percentWithin40%YSI')",
        "('[70, 180]', 'MBE')",
        "('[70, 180]', 'MARD%')",
        "('(180, 250]', 'percentWithin15%YSI')",
        "('(180, 250]', 'percentWithin20%YSI')",
        "('(180, 250]', 'percentWithin40%YSI')",
        "('(180, 250]', 'MBE')",
        "('(180, 250]', 'MARD%')",
        "('(250, 400]', 'percentWithin15%YSI')",
        "('(250, 400]', 'percentWithin20%YSI')",
        "('(250, 400]', 'percentWithin40%YSI')",
        "('(250, 400]', 'MBE')",
        "('(250, 400]', 'MARD%')",
        "('Beginning', 'MARD%')",
        "('Beginning', 'percentWithin15/15%YSI')",
        "('Beginning', 'percentWithin20/20%YSI')",
        "('Beginning', 'percentWithin40/40%YSI')",
        "('Middle', 'MARD%')",
        "('Middle', 'percentWithin15/15%YSI')",
        "('Middle', 'percentWithin20/20%YSI')",
        "('Middle', 'percentWithin40/40%YSI')",
        "('End', 'MARD%')",
        "('End', 'percentWithin15/15%YSI')",
        "('End', 'percentWithin20/20%YSI')",
        "('End', 'percentWithin40/40%YSI')"
     ]

    sim_df_columns = [
        'pump_bgs',
        'bg_actual',
        'bg_loop',
        'temp_basal',
        'insulin_relative_to_actual_basal',
        'carbLoop',
        'carbActual',
        'insulinLoop',
        'insulinActual',
        'cirLoop',
        'cirActual',
        'isfLoop',
        'isfActual',
        'sbrLoop',
        'sbrActual',
        'iob'
    ]

    for col in sim_df_columns:
        for sim_step in np.arange(0, 480, 5):
            sim_results_columns.append(col + "_" + str(sim_step))

    # Add seed true_bg trace
    for trace_num in np.arange(-14395, 5, 5):
        true_bg_col = 'seed_trueBG_{}'.format(trace_num)
        sim_results_columns.append(true_bg_col)

    # Add seed iCGM trace
    for trace_num in np.arange(-14395, 5, 5):
        iCGM_col = 'seed_iCGM_{}'.format(trace_num)
        sim_results_columns.append(iCGM_col)

    sim_results_template = pd.DataFrame(index=[0], columns=sim_results_columns)

    return sim_results_template


def get_slope(y):
    """
    Returns the least squares regression slope given a contiguous sequence y
    """

    # From SciPy lstsq usage Example Guide:
    # Rewrite y = mx + c equation as y = Ap
    # Where A = [[x 1]] and p = [[m], [c]]
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m/5  # Divide by the 5-min interval to get mg/dL/min resolution


def get_icgm_test_condition(icgm_last_30min):
    """Returns the test condition of the last 30 mintutes of icgm data

    Condition # || 30min Median BG (mg/dL) & 15min Rate of Change (mg/dL/min)
                ||
        1       ||   [40-70) & < -1
        2       ||   [70-180] & < -1
        3       ||   (180-400] & < -1
        4       ||   [40-70) & [-1 to 1]
        5       ||   [70-180] & [-1 to 1]
        6       ||   (180-400] & [-1 to 1]
        7       ||   [40-70) & > 1
        8       ||   [70-180] & > 1
        9       ||   (180-400] & > 1
    """

    median_bg = np.median(icgm_last_30min)
    slope = get_slope(icgm_last_30min[-3:])

    # Bound extreme iCGM to nearest min/max
    if median_bg < 40:
        median_bg = 40

    if median_bg > 400:
        median_bg = 400

    if (median_bg >= 40) & (median_bg < 70):
        if (slope < -1):
            icgm_test_condition = 1
        elif (slope >= -1) & (slope <= 1):
            icgm_test_condition = 4
        else:
            icgm_test_condition = 7

    elif (median_bg >= 70) & (median_bg <= 180):
        if (slope < -1):
            icgm_test_condition = 2
        elif (slope >= -1) & (slope <= 1):
            icgm_test_condition = 5
        else:
            icgm_test_condition = 8

    elif (median_bg > 180) & (median_bg <= 400):
        if (slope < -1):
            icgm_test_condition = 2
        elif (slope >= -1) & (slope <= 1):
            icgm_test_condition = 5
        else:
            icgm_test_condition = 8

    else:
        print(
                str(median_bg)
                + " and "
                + str(slope)
                + " do not match any test condition."
        )
        icgm_test_condition = 0

    return icgm_test_condition


def run_loop_scenario(
        sim_results_template,
        true_dataset_name,
        analysis_types,
        virtual_patient_num,
        bg_test_condition,
        sensor_num,
        true_bg_trace,
        icgm_traces,
        batch_icgm_results,
        all_sensor_properties,
        custom_table_df,
        simulation_duration_hours,
        df_settings,
        carb_ratios_original,
        isfs_original,
        basal_rates_original,
):

    sensor_properties = all_sensor_properties.loc[sensor_num-1, :]
    # icgm_trace = (
    #     np.round(icgm_traces[sensor_num-1]).astype(int).astype(str)
    # )
    icgm_trace = np.round(icgm_traces[sensor_num-1], 1).astype(str)

    # Re-generate the last 24 hours of iCGM data with a fresh sensor time
    starting_index = len(icgm_trace) - 288
    new_24hr_iCGM_trace = []

    sensor_noise_seed = (
            int(abs(sensor_properties['initial_bias']*1000000))
    )

    for t in range(288):
        # Get delayed true BG to feed into iCGM value generator
        delay_steps = int(sensor_properties['delay']/5)
        delayed_tBG_index = t + starting_index - delay_steps
        tBG_for_iCGM = true_bg_trace[delayed_tBG_index]

        # Generate iCGM value
        (
                new_iCGM_value,
                sensor_bias_factor,
                sensor_noise,
                sensor_drift_multiplier
        ) = sf.get_icgm_value(
                true_bg_value=tBG_for_iCGM,
                at_time=t*5,
                random_seed=sensor_noise_seed,
                initial_bias=(
                    sensor_properties['initial_bias']
                ),
                phi_drift=sensor_properties['phi_drift'],
                bias_drift_range=[
                    sensor_properties['bias_drift_range_start'],
                    sensor_properties['bias_drift_range_end']
                ],
                bias_drift_oscillations=(
                    sensor_properties['bias_drift_oscillations']
                ),
                bias_norm_factor=(
                    sensor_properties['bias_norm_factor']
                ),
                noise_coefficient=(
                    sensor_properties['noise_coefficient']
                )
            )

        new_24hr_iCGM_trace.append(new_iCGM_value)

    new_24hr_iCGM_trace = np.round(new_24hr_iCGM_trace, 1).astype(str)
    icgm_trace[-288:] = new_24hr_iCGM_trace

    icgm_last_30min = icgm_trace[-6:].astype(float)
    icgm_test_condition = get_icgm_test_condition(icgm_last_30min)
    # custom_table_df.loc['glucose_values', 1:2881] = icgm_trace

    # Truncate glucose values to just 24 hours up to evaluation point
    glucose_indices = [
        'glucose_dates',
        'glucose_values',
        'actual_blood_glucose',
        'glucose_units'
    ]

    glucose_section = custom_table_df.loc[glucose_indices].copy()

    truncated_glucose_section = (
        glucose_section[glucose_section.columns[-288:].insert(0, 'settings')]
    )

    custom_table_df = custom_table_df[custom_table_df.columns[:289]]
    custom_table_df.loc[glucose_indices] = truncated_glucose_section
    custom_table_df.loc['glucose_values', 1:289] = new_24hr_iCGM_trace

    all_sim_results = []

    for analysis_type in analysis_types:

        sim_results = sim_results_template.copy()

        if (analysis_type == 'tempBasal'):
            use_initial_recommended_bolus = False
            apply_meal_bolus_to_pump_sim = False

        if (analysis_type == 'correctionBolus'):
            use_initial_recommended_bolus = True
            apply_meal_bolus_to_pump_sim = False

        if (analysis_type == 'mealBolus'):
            use_initial_recommended_bolus = True
            apply_meal_bolus_to_pump_sim = True

            custom_table_df.loc['carb_values', '0'] = '30'
            custom_table_df.loc['actual_carbs', '0'] = '30'

        sim_df, scenario_results = (
            loop_sim.loop_simulator(
                custom_table_df,
                sensor_properties,
                simulation_duration_hours,
                use_initial_recommended_bolus,
                apply_meal_bolus_to_pump_sim
            )
        )

        # Drop last row (480) from sim_df
        sim_df = sim_df[:-1]

        # Append all results into simulation results dataframe
        sim_results['file_name'] = true_dataset_name
        sim_results['analysis_type'] = analysis_type
        sim_results['virtual_patient_num'] = virtual_patient_num
        sim_results['bg_test_condition'] = bg_test_condition
        sim_results['icgm_sensor_num'] = sensor_num
        sim_results['icgm_test_condition'] = icgm_test_condition
        sim_results['age'] = df_settings.loc['age', 'settings']
        sim_results['ylw'] = df_settings.loc['ylw', 'settings']
        sim_results['CIR'] = carb_ratios_original['actual_carb_ratios'][0]
        sim_results['ISF'] = isfs_original['actual_sensitivity_ratios'][0]
        sim_results['BR'] = basal_rates_original['actual_basal_rates'][0]
        sim_results['SEG_RS'] = (
            scenario_results.loc['SEG', 'loopRiskScore']
        )
        sim_results['pump_LBGI'] = (
            scenario_results.loc['LBGI', 'pumpValue']
        )
        sim_results['pump_LBGI_RS'] = (
            scenario_results.loc['LBGI', 'pumpRiskScore']
        )
        sim_results['pump_HBGI'] = (
            scenario_results.loc['HBGI', 'pumpValue']
        )
        sim_results['pump_HBGI_RS'] = (
            scenario_results.loc['HBGI', 'pumpRiskScore']
        )
        sim_results['loop_LBGI'] = (
            scenario_results.loc['LBGI', 'loopValue']
        )
        sim_results['loop_LBGI_RS'] = (
            scenario_results.loc['LBGI', 'loopRiskScore']
        )
        sim_results['loop_HBGI'] = (
            scenario_results.loc['HBGI', 'loopValue']
        )
        sim_results['loop_HBGI_RS'] = (
            scenario_results.loc['HBGI', 'loopRiskScore']
        )
        sim_results['DKAI'] = scenario_results.loc['DKAI', 'loopValue']
        sim_results['DKAI_RS'] = scenario_results.loc['DKAI', 'loopRiskScore']

        batch_results_indices = (
                pd.Series(list(batch_icgm_results.index)).astype(str).values
        )
        sim_results[batch_results_indices] = (
                batch_icgm_results.icgmSensorResults.values
        )

        sensor_properties_indices = (
                'sensor_'
                + pd.Series(list(sensor_properties.index)).astype(str).values
        )

        sim_results[sensor_properties_indices] = (
            sensor_properties.values
        )

        sim_results['sensor_bias_factor'] = sensor_bias_factor
        sim_results['sensor_noise_seed'] = sensor_noise_seed
        sim_results['sensor_noise'] = str(sensor_noise)
        sim_results['sensor_drift_multiplier'] = str(sensor_drift_multiplier)

        # Add the flattened sim_df to the sim_results
        flattened_sim = sim_df.values.flatten('F')

        sim_columns = []
        for col_name in sim_df.columns:
            cols = pd.Series(col_name + '_' + sim_df.index.astype(str).values)
            sim_columns.append(cols)

        sim_columns = pd.concat(sim_columns).values

        sim_results[sim_columns] = flattened_sim

        # Add true_bg sensor seed trace
        seed_tBG_columns = []
        for trace_num in np.arange(-14395, 5, 5):
            true_bg_col = 'seed_trueBG_{}'.format(trace_num)
            seed_tBG_columns.append(true_bg_col)

        sim_results[seed_tBG_columns] = true_bg_trace

        # Add seed iCGM trace
        seed_iCGM_columns = []
        for trace_num in np.arange(-14395, 5, 5):
            iCGM_col = 'seed_iCGM_{}'.format(trace_num)
            seed_iCGM_columns.append(iCGM_col)

        sim_results[seed_iCGM_columns] = icgm_trace

        all_sim_results.append(sim_results)

        print(".", end="")

    all_sim_results = pd.concat(all_sim_results).reset_index(drop=True)

    return all_sim_results


def scenario_risk_assessment(
        virtual_patient_num,
        bg_test_condition,
        analysis_types,
        true_dataset_name,
        scenario_folder,
        n_sensors,
        simulation_duration_hours
):
    """A complete risk assessment for one scenario (BG Test Condition)"""

    # Create simulation results template
    sim_results_template = create_results_template()

    # Import Scenario
    print(
        "\nSTARTING: \n"
        + "Virtual Patient # " + str(virtual_patient_num) + "\n"
        + "Test Condition # " + str(bg_test_condition) + "\n"
        + "File: " + true_dataset_name
    )

    table_path_name = os.path.join(
        scenario_folder,
        true_dataset_name
    )
    custom_table_df = pd.read_csv(table_path_name, index_col=0)
    inputs_from_file = sf.input_table_to_dict(custom_table_df)

    # Convert inputs to dataframes
    (
        basal_rates_original,
        carb_events_original,
        carb_ratios_original,
        dose_events_original,
        cgm_df_original,
        df_last_temporary_basal,
        df_misc,
        isfs_original,
        df_settings,
        df_target_range_original
    ) = sf.dict_inputs_to_dataframes(inputs_from_file)

    # Isolate the 10-day True BG (tBG)
    true_bg_trace = np.array(cgm_df_original["actual_blood_glucose"])

    # Generate iCGM Sensors and Traces for scenario using the tBG
    print("Creating iCGM Sensors...", end="")
    (icgm_traces, all_sensor_properties, batch_icgm_results) = (
        icgm_sim.icgm_simulator(
            n_sensors=n_sensors,
            true_bg_trace=true_bg_trace,
            true_dataset_name=true_dataset_name
        )
    )
    print("done!")

    # Create an array to store all icgm trace results
    # Feed each iCGM trace into PyLoopKit
    print("Simulating each iCGM Trace and analysis type in PyLoopKit", end="")
    individual_sensor_results = []

    # Startup CPU multiprocessing pool
    loop_multiprocess_pool = Pool(os.cpu_count())

    loop_results_array = [
        loop_multiprocess_pool.apply_async(
            run_loop_scenario,
            args=[
                    sim_results_template,
                    true_dataset_name,
                    analysis_types,
                    virtual_patient_num,
                    bg_test_condition,
                    sensor_num,
                    true_bg_trace,
                    icgm_traces,
                    batch_icgm_results,
                    all_sensor_properties,
                    custom_table_df,
                    simulation_duration_hours,
                    df_settings,
                    carb_ratios_original,
                    isfs_original,
                    basal_rates_original,

            ]
        ) for sensor_num in range(1, len(icgm_traces)+1)
    ]

    loop_multiprocess_pool.close()
    loop_multiprocess_pool.join()

    individual_sensor_results = []

    for result_loc in range(len(loop_results_array)):
        try:
            individual_sensor_results.append(
                loop_results_array[result_loc].get()
            )
        except Exception as e:
            print('Failed to get results! ' + str(e))
            exception_text = traceback.format_exception(*sys.exc_info())
            print('\nException Text:\n')
            for text_string in exception_text:
                print(text_string)

    print("done!")

    scenario_results_df = (
        pd.concat(individual_sensor_results).reset_index(drop=True)
    )

    # Combine scenario and iCGM sensor data into a original single dataframe
    # for sensor_num in range(len(icgm_traces)):
    #     trace_index_name = 'sensor_' + str(sensor_num) + '_iCGM_trace'
    #     custom_table_df.loc[trace_index_name, 1:2881] = icgm_traces[sensor_num]

    #     icgm_trace = icgm_traces[sensor_num]
    #     sensor_properties = all_sensor_properties.loc[sensor_num]

    #     for col in sensor_properties.columns:
    #         settings_name = 'sensor_' + str(sensor_num) + '_' + col
    #         custom_table_df.loc[settings_name, 'settings'] = (
    #             sensor_properties[col].values
    #         )

    # Save full dataframe and iCGM sensor results to new .csv file

    return scenario_results_df


def main(scenario_folder, n_sensors):
    """Collects and aggregates metadata from the pipeline for each:
        * (A)nalysis Type
        * (V)irtual Patient
        * (B)G Test Condtion
    """

    # Optional iCGM settings
    # SPECIAL_CONTROLS_CRITERIA_THRESHOLDS=[0.85,   # A
    #                                       0.70,   # B
    #                                       0.80,   # C
    #                                       0.98,   # D
    #                                       0.99,   # E
    #                                       0.99,   # F
    #                                       0.87],  # G
    # use_g6_accuracy_in_loss=False
    # bias_type="percentage_of_value"
    # bias_drift_type="random"
    # random_seed=0
    # verbose=False
    # save_results=False
    # make_figures=False

    simulation_duration_hours = 8

    scenario_file_names = os.listdir(scenario_folder)

    virtual_patients = (
        pd.Series(
            scenario_file_names
        ).str.split('_condition', expand=True)[0].unique()
    )

    # Setup all simulation combination parameters
    analysis_types = ['tempBasal', 'correctionBolus', 'mealBolus']
    virtual_patient_numbers = np.arange(1, len(virtual_patients)+1)
    bg_test_conditions = np.arange(1, 9+1)

    all_simulation_results = []

    # SERIAL PROCESS VERSION - LOOP THROUGH ALL Patients and conditions

    for virtual_patient_num in virtual_patient_numbers:
        for bg_test_condition in bg_test_conditions:

            true_dataset_name = (
                virtual_patients[virtual_patient_num-1]
                + '_condition'
                + str(bg_test_condition)
                + '.csv'
            )

            if true_dataset_name in scenario_file_names:

                sim_results = (
                    scenario_risk_assessment(
                        virtual_patient_num,
                        bg_test_condition,
                        analysis_types,
                        true_dataset_name,
                        scenario_folder,
                        n_sensors,
                        simulation_duration_hours
                    )
                )

                all_simulation_results.append(sim_results)

    print("done!")

    all_simulation_results = (
        pd.concat(all_simulation_results).reset_index(drop=True)
    )

    return all_simulation_results


# %%
if __name__ == "__main__":

    # Scenario file locations
    scenario_folder = "../data/interim/sample-snapshot-export/"
    results_folder = "../data/processed/"

    # Number of sensors to create per scenario
    n_sensors = 3

    start_time = time.time()

    all_simulation_results = main(scenario_folder, n_sensors)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time)/60
    elapsed_time_message = (
        "All simulations completed in: "
        + str(elapsed_minutes)
        + " minutes\n"
    )
    print(elapsed_time_message)

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    results_export_filename = 'sample-risk-sim-results.gz'
    results_path = results_folder + results_export_filename

    all_simulation_results.to_csv(results_path,
                                  index=False,
                                  compression='gzip')
