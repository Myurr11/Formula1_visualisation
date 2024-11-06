import os
import pickle as pickle
import fastf1 as ff1
import numpy as np
import pandas as pd


def downcast_df(df):
    for column in df:
        if df[column].dtype == 'float64':
            df[column] = pd.to_numeric(df[column], downcast='float')
        if df[column].dtype == 'int64':
            df[column] = pd.to_numeric(df[column], downcast='integer')

    return df


def get_lap_data(season, gp, session_in, cache_path=None, save_path=None):

    # Use cached data from API calls (or don't)
    if cache_path:
        ff1.Cache.enable_cache(cache_path)

    # Load the session data
    session = ff1.get_session(season, gp, session_in)
    session.load()

    # Get laps data (excl. formation lap)
    laps = session.laps
    laps = laps.loc[laps['LapNumber'] >= 1]

    # Add final driver position to laps data (for future processing convenience)
    if session_in == 'Sprint':
        results = laps.sort_values(by='LapNumber', ascending=False).drop_duplicates(['DriverNumber'])
        results.sort_values(by=['LapNumber', 'Time'], ascending=[False, True], inplace=True, ignore_index=True)
        results['Position'] = results.index+1
        results = results[['DriverNumber', 'Position']]
    else:
        results = pd.DataFrame({'DriverNumber': session.results.DriverNumber, 'Position': session.results.Position})
    laps = laps.merge(results, how='left', on='DriverNumber')

    # Get winner's delta to each driver
    laps_winner = laps.loc[laps['Position'] == 1]
    laps_winner = laps_winner[['Time', 'LapNumber']]
    laps_winner.columns = ['WinnerTime', 'LapNumber']
    laps = laps.merge(laps_winner, how='left', on='LapNumber')
    laps['DeltaWinner'] = (laps['Time'] - laps['WinnerTime']) / np.timedelta64(1, 's')

    # Identify the pit stops (for future processing convenience)
    laps.loc[:, 'PitStop'] = ~laps['PitInTime'].isnull()

    # Parse the track status into separate columns (for future processing convenience)
    laps.loc[:, 'Yellow'] = laps['TrackStatus'].str.contains('2')
    laps.loc[:, 'Red'] = laps['TrackStatus'].str.contains('5')
    laps.loc[:, 'Safety'] = laps['TrackStatus'].str.contains('4')
    laps.loc[:, 'Virtual'] = laps['TrackStatus'].str.contains('6')
    laps.loc[:, 'Nominal'] = (~laps['Red']) & (~laps['Yellow']) & (~laps['Safety']) & (~laps['Virtual'])

    # Store relevant data as a dataframe; downcast ints/floats where possible to reduce size
    laps = laps[['Driver', 'LapNumber', 'LapTime', 'Position', 'DeltaWinner', 'PitInTime', 'PitStop', 'Time', 'Stint',
                 'Compound', 'Yellow', 'Red', 'Safety', 'Virtual', 'Nominal']]
    laps_df = downcast_df(pd.DataFrame(laps))

    # Save dataframe if requested
    if save_path is not None:
        laps_df.to_pickle(save_path)

    return laps_df


def get_telemetry_data(season, gp, session, cache_path=None, save_path=None, downsample=1):
    # Use cached data from API calls (or don't)
    if cache_path:
        ff1.Cache.enable_cache(cache_path)

    # Load the session data
    session = ff1.get_session(season, gp, session)
    session.load()

    # Get laps data (excl. formation lap)
    laps = session.laps
    laps = laps.loc[laps['LapNumber'] >= 1]

    # Get all drivers
    drivers = pd.unique(laps['Driver'])

    # Retrieve each driver's telemetry, resetting distance to 0 at start of each lap
    # (based on https://medium.com/towards-formula-1-analysis/formula-1-data-analysis-tutorial-2021-russian-gp-to-box-
    # or-not-to-box-da6399bd4a39)
    telemetry = pd.DataFrame()
    for driver in drivers:
        driver_laps = laps.pick_driver(driver)
        print("Retrieving telemetry data for " + driver)

        for lap in driver_laps.iterlaps():
            try:
                driver_telemetry = lap[1].get_telemetry()
            except Exception as err:
                print("Error accessing data; skipping and continuing")
                pass
            else:
                driver_telemetry['Driver'] = driver
                driver_telemetry['LapNumber'] = lap[1]['LapNumber']
                driver_telemetry['Compound'] = lap[1]['Compound']
                driver_telemetry['Brake'] = driver_telemetry['Brake'].astype(int) * 100

                telemetry = telemetry.append(driver_telemetry)

    # Store relevant data as a dataframe; downsample if requested; downcast ints/floats where possible to reduce size
    telemetry = telemetry[['Driver', 'X', 'Y', 'Speed', 'nGear', 'Throttle', 'Brake', 'Distance', 'LapNumber']]
    telemetry = telemetry.iloc[::downsample]
    telemetry_df = downcast_df(pd.DataFrame(telemetry))

    # Save dataframe if requested
    if save_path:
        telemetry_df.to_pickle(save_path)

    return telemetry_df


def add_session_to_site_data(season, gp, session, path_to_data, cache_path=None, downsample=1):
    # Create a folder for the year if it doesn't exist already
    path_to_year = os.path.join(path_to_data, str(season))
    if not os.path.exists(path_to_year):
        os.makedirs(path_to_year)

    # Pull in the existing data (if relevant)
    path_to_gp_data = os.path.join(path_to_year, gp.replace(" ", "_") + '.pickle')
    if os.path.isfile(path_to_gp_data):
        with open(path_to_gp_data, 'rb') as handle:
            gp_data = pickle.load(handle)
    else:
        gp_data = {'lap_data': {}, 'telemetry_data': {}}

    path_to_drop_down_data = os.path.join(path_to_data, 'drop_down_data.pickle')
    if os.path.isfile(path_to_drop_down_data):
        with open(path_to_drop_down_data, 'rb') as handle:
            drop_down_data = pickle.load(handle)
    else:
        drop_down_data = {}

    # Add (or overwrite) lap and telemetry data for the requested gp and session
    laps = get_lap_data(season, gp, session, cache_path=cache_path)
    gp_data['lap_data'][session] = laps

    telemetry = get_telemetry_data(season, gp, session, cache_path=cache_path, downsample=downsample)
    gp_data['telemetry_data'][session] = telemetry

    # Add (or overwrite) drop down options for the requested gp and session
    if str(season) not in drop_down_data.keys():
        drop_down_data[str(season)] = {}
    if str(gp) not in drop_down_data[str(season)].keys():
        drop_down_data[str(season)][gp] = {}
    if session not in drop_down_data[str(season)][gp].keys():
        drop_down_data[str(season)][gp][session] = {}

    for driver in telemetry['Driver'].unique():
        drop_down_data[str(season)][gp][session][driver] = telemetry[telemetry['Driver'] == driver][
            'LapNumber'].unique()

    # Save as pickle
    with open(path_to_gp_data, 'wb') as handle:
        pickle.dump(gp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path_to_drop_down_data, 'wb') as handle:
        pickle.dump(drop_down_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
