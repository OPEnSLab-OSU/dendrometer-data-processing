import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import linecache


class DataFormatter:
    def __init__(self) -> None:
        self.time_offset = 0
        print("Constructor")

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

    def get_time_differences(self, folders: dict):
        times = []
        for files in folders.values():
            for file in files:
                first_entry = linecache.getline(Path.as_posix(file), 3)
                linecache.clearcache()

                time_entry = first_entry.split(",")[2:4]
                cur_time = pd.to_datetime(time_entry[0] + " " + time_entry[1])
                print(cur_time)
                times.append(cur_time)

        return times

    def l1_formatting(self, folders: dict):
        for folder in folders.keys():
            print(folder)
            for file in folders[folder]:
                df = pd.read_csv(Path.as_posix(file), header=[0, 1])
                df = self._initial_formatting(df)
                df = self._fix_timestamp(df)
                df = self.adjust_flow(df)

                self.save_plot(df, file)

    def _initial_formatting(self, df):
        # Change column names
        df = df.rename(
            columns={'Unnamed: 1_level_0': 'ID', 'Unnamed: 3_level_0': 'Timestamp', 'Unnamed: 7_level_0': 'SHT31D'})

        # Drop last column because it is empty
        df = df.iloc[:, :-1]
        return df

    def _fix_timestamp(self, df):
        date_time_combined = pd.to_datetime(
            df[("Timestamp", "date")] +
            ' ' +
            df[("Timestamp", "time")])

        df.drop(columns=[
                ("Timestamp", "date"),
                ("Timestamp", "time")], inplace=True)

        df.insert(2, "Time", date_time_combined)
        return df

    def adjust_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        prev_idx, prev_serial = 0, df.iloc[0][('AS5311', 'Serial_Value')]
        initial, wrap = df.iloc[0][('AS5311', 'Serial_Value')], 0
        df.at[0, 'Calculated'] = 0

        cur_idx = 1
        while cur_idx < df.shape[0]:
            cur_serial = df.iloc[cur_idx][('AS5311', 'Serial_Value')]

            change = (cur_serial - prev_serial)/(cur_idx - prev_idx)
            df.at[cur_idx, 'Change'] = change

            calculated_serial_data = prev_serial + wrap - initial

            if abs(change) > 2000:
                # Wrap up
                if change < 0:
                    print("-- Wrapped up", prev_serial, cur_serial, change)
                    wrap += 4095

                # Wrap down
                elif change > 0:
                    print("-- Wrapped down", prev_serial, cur_serial, change)
                    wrap -= 4095

            # Calculate the displacement data
            calculated_serial_data = cur_serial + wrap - initial

            # Data to use in the plot
            df.at[cur_idx, 'Calculated'] = calculated_serial_data

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
    formatter.get_time_differences(files)

    formatter.l1_formatting(files)

if __name__ == "__main__":
    main()
