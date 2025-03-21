import pandas as pd
import os

file_path = './Synched_Data_GR0_22_DEN_MAXZ1_25/'
file_date = ['101922', '102122', '111422', '111622', '120522', '120722', 
             '013023', '020123', '031323', '031523', '041723', '041923', '061523']

read_co_dict = {}
read_ag_1_dict = {}

for date in file_date:
    file_name_1 = f'COTALK/DAYCOTALK_TYPE_{date}_COTALK0_22_DEN_012325_V2264433238.csv'
    file_name_2 = f'PAIRANGLES/DAILY_ANGLES{date}_GR0_22_DEN_012325_V2264433238.csv'
    full_path_1 = os.path.join(file_path, file_name_1)
    full_path_2 = os.path.join(file_path, file_name_2)

    # Load and process COTALK data
    header_co = pd.read_csv(full_path_1)
    read_co = header_co[["SUBJECTID", "TIME", "KC_X", "KC_Y"]].copy()  # Ensure a copy
    read_co["TIME"] = pd.to_datetime(read_co["TIME"])
    read_co.set_index(["SUBJECTID", "TIME"], inplace=True)  # Multi-index

    resampled_co = (
        read_co
        .resample('1s', level="TIME")  # Resample at the 'TIME' level
        .first()
        .dropna()
        .reset_index()  # Restore index properly
    )

resampled_co["Formatted_TIME"] = resampled_co["TIME"].dt.strftime('%H:%M:%S')

read_co_dict[date] = resampled_co  # Store processed DataFrame


    if not os.path.exists(full_path_1) or not os.path.exists(full_path_2):
        print(f"Skipping {date}: File not found")
        continue

    header_co = pd.read_csv(full_path_1)
    read_co = header_co[["SUBJECTID", "TIME", "KC_X", "KC_Y"]].copy()
    read_co["TIME"] = pd.to_datetime(read_co["TIME"])
    read_co.set_index("TIME", inplace=True)

    resampled_co = (
        read_co.groupby("SUBJECTID")
        .resample('1s')
        .first()
        .dropna()
        .reset_index()
    )
    resampled_co["Formatted_TIME"] = resampled_co["TIME"].dt.strftime('%H:%M:%S')

    read_co_dict[date] = resampled_co 

    header_ag = pd.read_csv(full_path_2)
    header_ag.columns = header_ag.columns.str.strip()

    read_ag_1 = header_ag[['Person 1', 'Interaction Time', 'Interaction Millisecond', 'Angle1', 'Leftx', 'Lefty', 'Rightx', 'Righty']].copy()
    
    read_ag_1_unique = read_ag_1.drop_duplicates(
        subset=['Person 1', 'Interaction Time', 'Interaction Millisecond', 'Leftx', 'Lefty', 'Rightx', 'Righty'],
        keep='first'
    ).copy()

    read_ag_1_unique['Interaction Time'] = pd.to_datetime(read_ag_1_unique['Interaction Time'])
    read_ag_1_unique['X'] = (read_ag_1_unique['Leftx'] + read_ag_1_unique['Rightx']) / 2
    read_ag_1_unique['Y'] = (read_ag_1_unique['Lefty'] + read_ag_1_unique['Righty']) / 2
    read_ag_1_unique.drop(columns=['Leftx', 'Lefty', 'Rightx', 'Righty'], inplace=True)

    read_ag_1_sorted = read_ag_1_unique.sort_values(by=['Person 1', 'Interaction Time', 'Interaction Millisecond'])

    read_ag_1_dict[date] = read_ag_1_sorted