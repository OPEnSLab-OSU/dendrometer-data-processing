import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from plotter import Plotter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import numpy as np
from enum import Enum

class Status(Enum):
    GREEN   = "Green"
    YELLOW  = "Yellow"
    RED     = "Red"
    ERROR   = "Error"

DEFAULT_FILE_SIZE = 100


class DataFormatter:
    def __init__(self) -> None:
        self.deployment_time_map = {}

    
    def load_deployment_time(self) -> dict:
        """
        Funtion used to correct the time column from csv files.
        Depends on a file called deployment_time.csv to get the correct initial timings.
        """
        
        cur_path = Path.cwd()
        timetable_file = Path.joinpath(cur_path, "data", "deployment_time.csv")
        deployment_time_df = pd.read_csv(timetable_file)
        deployment_time_df = deployment_time_df.fillna("")

        for index, row in deployment_time_df.iterrows():
            if not row["Start Date"] or not row["Start Time"]:
                continue
            timestamp = pd.to_datetime(row["Start Date"] +
                                       ' ' +
                                       row["Start Time"])

            timestamp = timestamp.tz_localize(None)
            self.deployment_time_map[str(row["Device ID"])] = timestamp
            

    def find_valid_files(self, min_file_size: int = DEFAULT_FILE_SIZE) -> dict:
        """
        Find the valid data files inside the data folder

        min_file_size -- the minimum valid size for a file in bytes (default 1000000 bytes)
        """

        cur_path = Path.cwd()
        data_path = Path.joinpath(cur_path, "data")
        folders = [folder for folder in data_path.iterdir() if folder.is_dir()]
        folder_file_map = defaultdict(list)

        for folder in folders:
            files_path = Path.joinpath(data_path, folder)
            files = [file for file in Path(files_path).glob("*.csv")]

            for file in files:
                file_size = Path(Path.joinpath(
                    files_path, file)).stat().st_size

                if file_size > min_file_size:
                    folder_file_map[folder].append(file)

        return folder_file_map

    
    def _initial_formatting(self, df):
        # Change column names
        df = df.rename(
            columns={'Unnamed: 1_level_0': 'ID', 'Unnamed: 3_level_0': 'timestamp', 
                     'Unnamed: 6_level_0': 'Analog', 'Unnamed: 8_level_0': 'SHT31', 'Unnamed: 9_level_0': 'SHT31',
                     'Unnamed: 11_level_0': 'AS5311', 'Unnamed: 12_level_0': 'AS5311', 'Unnamed: 13_level_0': 'AS5311',
                     'Unnamed: 14_level_0': 'AS5311'})

        df.columns = df.columns.map('_'.join)

        # Drop last column because it is empty
        df = df.iloc[:, :-1]
        
        df['timestamp_time_local'] = pd.to_datetime(df['timestamp_time_local'])
        # df.set_index(['timestamp_time_local'], inplace=True)
                
        return df
    
    
    def calculate_change(self, df: pd.DataFrame):
        df['change'] = df['AS5311_pos_raw'].diff()
        # print(df)
        return df
    
    def categorize_color(self, row):
        if row['AS5311_Alignment'] == "Green":
            return "Green"
        elif row['AS5311_Alignment'] == "Yellow":
            return "Yellow"
        elif row['AS5311_Alignment'] == "Red":
            return "Red"
        elif row['AS5311_Alignment'] == 'Error':
            return "Gray"
    
    def categorize_status(self, row, status: Status):
        if status.value == row["AS5311_Alignment"]:
            return row["AS5311_pos_avg"]
        return np.nan
    
    def color_graph(self, df: pd.DataFrame):
        fig = go.Figure()
        x = df["timestamp_time_local"]
        y = df["AS5311_pos_avg"]
        # df['color'] = df.apply(lambda row: self.categorize_color(row), axis=1)
        
        df["AS5311_pos_avg_GREEN"]  = df.apply(lambda row: self.categorize_status(row, Status.GREEN), axis=1)
        df["AS5311_pos_avg_YELLOW"] = df.apply(lambda row: self.categorize_status(row, Status.YELLOW), axis=1)
        df["AS5311_pos_avg_RED"]    = df.apply(lambda row: self.categorize_status(row, Status.RED), axis=1)
        df["AS5311_pos_avg_ERROR"]  = df.apply(lambda row: self.categorize_status(row, Status.ERROR), axis=1)
        
        # fig = px.line(df, x=x, y=["AS5311_pos_avg_GREEN", "AS5311_pos_avg_YELLOW", "AS5311_pos_avg_RED", "AS5311_pos_avg_ERROR"],
        #               labels={
        #               "x": "Date",
        #               "value": "Serial Value",
        #               "variable": "Alignment"
        #               },
        #               title="Time v Displacement")
        
        fig.add_trace(go.Scatter(x=df["timestamp_time_local"], y=df["AS5311_pos_avg_GREEN"], name=Status.GREEN.name))
        fig.add_trace(go.Scatter(x=df["timestamp_time_local"], y=df["AS5311_pos_avg_YELLOW"], name=Status.YELLOW.name))
        fig.add_trace(go.Scatter(x=df["timestamp_time_local"], y=df["AS5311_pos_avg_RED"], name=Status.RED.name))
        fig.add_trace(go.Scatter(x=df["timestamp_time_local"], y=df["AS5311_pos_avg_ERROR"], name=Status.ERROR.name))
        
        
        fig.update_layout(title="Time vs Displacement", xaxis_title="Date", yaxis_title="Serial Value", legend_title="Status color")
        
        fig.update_traces(connectgaps=False)
        fig.show()
        print(df)
    

    
    def adjust_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_idx, prev_serial = 0, df.iloc[0]['AS5311_pos_raw']
        initial, wrap = df.iloc[0]['AS5311_pos_raw'], 0
        # df.at[0, 'Calculated Serial'] = 0
        
        # print("Number of rows: ", df.shape[0])
        # print("SD: ", df[('AS5311', 'pos_raw')].std())
        # print(df.describe())

        # print(df.columns)
        
        # cur_idx = 1
        # while cur_idx < df.shape[0]:
        #     cur_serial = df.iloc[cur_idx]['AS5311_pos_raw']

        #     change = (cur_serial - prev_serial)/(cur_idx - prev_idx)
        #     df.at[cur_idx, 'Change'] = change

        #     calculated_serial_data = prev_serial + wrap - initial

        #     if abs(change) > 2000:
        #         # Wrap up
        #         if change < 0:
        #             print("  -- Wrapped up", prev_serial, cur_serial, change)
        #             wrap += 4095

        #         # Wrap down
        #         elif change > 0:
        #             print("  -- Wrapped down", prev_serial, cur_serial, change)
        #             wrap -= 4095

        #     # Calculate the displacement data
        #     calculated_serial_data = cur_serial + wrap - initial

        #     # Data to use in the plot
        #     df.at[cur_idx, 'Calculated Serial'] = calculated_serial_data

        #     prev_serial = cur_serial
        #     prev_idx = cur_idx
        #     cur_idx += 1


        # fig = px.histogram(df, x='Change')
        # fig.show()
        return df

    def l1_formatting(self, folders: dict):
        dfs = {}  # Dataframes
        # self.load_deployment_time()

        for folder in folders.keys():
            print("-------------------------------------------")
            print(folder)
            for file in folders[folder]:
                dendrometer_id = str(file).split("/")[-2]

                df = pd.read_csv(Path.as_posix(file), header=[0, 1])
                df = self._initial_formatting(df)
                df = self.adjust_flow(df)
                df = self.calculate_change(df)
                self.color_graph(df)
    
                dfs[dendrometer_id] = (file, df.copy())
                # print(df.head())
                # print(df)
            print("-------------------------------------------")
        

        plotter = Plotter()
        # plotter.load_deployment_map()
        # pair_mapping = plotter.get_pair_mapping()

        for _, (filename, df) in dfs.items():
            plotter.save_plot(filename, df)
            # plotter.save_plot_vpd(filename, df)
            plotter.save_csv(filename, df)

        # for pair in pair_mapping.values():
        #     dend1, dend2 = pair
        #     dend1, dend2 = str(dend1), str(dend2)
        # 
            # if dend1 in dfs and dend2 in dfs:
            #     plotter.save_plot_pair(dfs[str(dend1)], dfs[str(dend2)])


def main():
    formatter = DataFormatter()
    files = formatter.find_valid_files()
    formatter.l1_formatting(files)


if __name__ == "__main__":
    main()
