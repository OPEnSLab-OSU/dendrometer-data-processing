import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import linecache
from plotter import Plotter


class DataFormatter:
    def __init__(self) -> None:
        self.deployment_time_map = {}
        print("Constructor")

    def load_deployment_time(self) -> dict:
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

    def find_valid_files(self, min_file_size: int = 1000000) -> dict:
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

    def get_time_delta(self, df: pd.DataFrame, deploy_time: pd.Timestamp):
        # print(type(df.iloc[0]["Time"].item()))
        print("-", deploy_time, df.iloc[0]["Time"].item())
        print("- Time delta:", pd.Timedelta(
            deploy_time - df.iloc[0]["Time"].item()))
        return pd.Timedelta(deploy_time - df.iloc[0]["Time"].item())

    def l1_formatting(self, folders: dict):
        dfs = {}  # Dataframes
        self.load_deployment_time()

        for folder in folders.keys():
            print("-------------------------------------------")
            print(folder)
            for file in folders[folder]:
                dendrometer_id = str(file).split("/")[-2]

                df = pd.read_csv(Path.as_posix(file), header=[0, 1])
                df = self._initial_formatting(df)
                df = self._fix_timestamp(df, dendrometer_id)
                df = self.adjust_flow(df)

                dfs[dendrometer_id] = (file, df.copy())
                # print(df.head())
                # print(type(df["VPD"].values.flatten()),
                #       df["VPD"].values.flatten())
            print("-------------------------------------------")

        plotter = Plotter()
        plotter.load_deployment_map()
        pair_mapping = plotter.get_pair_mapping()

        for _, (filename, df) in dfs.items():
            plotter.save_plot(filename, df)
            plotter.save_plot_vpd(filename, df)

        for pair in pair_mapping.values():
            dend1, dend2 = pair
            dend1, dend2 = str(dend1), str(dend2)

            if dend1 in dfs and dend2 in dfs:
                plotter.save_plot_pair(dfs[str(dend1)], dfs[str(dend2)])

    def _initial_formatting(self, df):
        # Change column names
        df = df.rename(
            columns={'Unnamed: 1_level_0': 'ID', 'Unnamed: 3_level_0': 'Timestamp', 'Unnamed: 7_level_0': 'SHT31D'})

        # Drop last column because it is empty
        df = df.iloc[:, :-1]
        return df

    def _fix_timestamp(self, df: pd.DataFrame, dendrometer_id):
        date_time_combined = pd.to_datetime(
            df[("Timestamp", "date")] +
            ' ' +
            df[("Timestamp", "time")])

        df.drop(columns=[
                ("Timestamp", "date"),
                ("Timestamp", "time")], inplace=True)

        df.insert(2, "Time", date_time_combined)

        """
        dt.tz_localize(tz="GMT"): assigns the current time stamp to be in UTC
        dt.tz_convert(tz="America/Los_Angeles"): converts UTC to Pacific Daylight Time (Data is from summer)
        dt.tz_localize(None): strips it back down to naive timestamp so it doesn't have the -7 at the end
        """
        df["Time"] = pd.to_datetime(df["Time"])  \
                       .dt.tz_localize(tz="UTC") \
                       .dt.tz_convert(tz="America/Los_Angeles") \
                       .dt.tz_localize(None)

        if dendrometer_id in self.deployment_time_map:
            time_delta = self.get_time_delta(
                df,
                self.deployment_time_map[dendrometer_id]
            )

            df.insert(3, "Adjusted Time", df["Time"] + time_delta)
        return df

    def adjust_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_idx, prev_serial = 0, df.iloc[0][('AS5311', 'Serial_Value')]
        initial, wrap = df.iloc[0][('AS5311', 'Serial_Value')], 0
        df.at[0, 'Calculated Serial'] = 0

        cur_idx = 1
        while cur_idx < df.shape[0]:
            cur_serial = df.iloc[cur_idx][('AS5311', 'Serial_Value')]

            change = (cur_serial - prev_serial)/(cur_idx - prev_idx)
            df.at[cur_idx, 'Change'] = change

            calculated_serial_data = prev_serial + wrap - initial

            if abs(change) > 2000:
                # Wrap up
                if change < 0:
                    print("  -- Wrapped up", prev_serial, cur_serial, change)
                    wrap += 4095

                # Wrap down
                elif change > 0:
                    print("  -- Wrapped down", prev_serial, cur_serial, change)
                    wrap -= 4095

            # Calculate the displacement data
            calculated_serial_data = cur_serial + wrap - initial

            # Data to use in the plot
            df.at[cur_idx, 'Calculated Serial'] = calculated_serial_data

            prev_serial = cur_serial
            prev_idx = cur_idx
            cur_idx += 1

        return df

    def save_plot(self, df: pd.DataFrame, filename) -> None:
        filename = filename.with_suffix('')
        dend_file_name = str(filename).split("/")[-2:]
        plot_title = f"Dendrometer_{dend_file_name[0]}_{dend_file_name[1]}"

        plt.figure(dpi=600, figsize=(11.69, 8.27))
        fig1 = plt.subplot()
        fig1.set_title(plot_title)
        fig1.plot(df["Time"], df[('AS5311', 'Serial_Value')])
        fig1.plot(df["Time"], df['Calculated'])
        fig1.legend(['Raw data', 'Over/Under flow adjusted'])
        fig1.set_xlabel("Time")
        fig1.set_ylabel("Displacement (serial value)")
        plt.savefig(f"data/{dend_file_name[0]}/{dend_file_name[1]}.pdf")


def main():
    formatter = DataFormatter()
    files = formatter.find_valid_files()
    formatter.l1_formatting(files)


if __name__ == "__main__":
    main()
