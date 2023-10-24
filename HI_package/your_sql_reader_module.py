import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

class SQLDataReader:

    def __init__(self, sql_db_path: str):
        if not sql_db_path.endswith('.sqlite'):
            raise ValueError('The path must be a .sqlite file')

        self._sql_db_path = sql_db_path
        self._connection = self._create_connection()
        self.table_list = self._extract_table_list()

    def get_data_pandas(self, table_name: str = 'data', start: str = None, end: str = None) -> tuple:

        if start:
            self._check_and_convert_datetime_string(start)
        if end:
            self._check_and_convert_datetime_string(end)

        if table_name not in self.table_list:
            return pd.DataFrame({}), {}

        data = pd.read_sql_query(f"SELECT * FROM {table_name}", self._connection, index_col='id')
        data, unit = self._process_data(data, start, end)

        data.timestamp = pd.to_datetime(data.timestamp)
        data.drop_duplicates(subset='timestamp', inplace=True)

        data.index = np.arange(len(data.index))

        return data, unit

    def get_raw_data_pandas(self, table_name: str = 'data', start: str = None, end: str = None) -> pd.DataFrame:

        if start:
            self._check_and_convert_datetime_string(start)
        if end:
            self._check_and_convert_datetime_string(end)

        if table_name not in self.table_list:
            return pd.DataFrame({})

        data = pd.read_sql_query(f"SELECT * FROM {table_name}", self._connection, index_col='id')
        data = self._filter_data_by_time(data, start, end)
        data.index = np.arange(len(data.index))

        return data

    def _process_data(self, data, start, end):
        data['timestamp_[iso]'] = pd.to_datetime(data['timestamp_[iso]']).dt.round('2min')
        data = self._filter_data_by_time(data, start, end)
        #         data['timestamp_[iso]'] = (data['timestamp_[iso]']).dt.isoformat()
        data, unit = self._separate_data_unit(data)
        return data, unit

    def _filter_data_by_time(self, data, start, end):
        if start:
            data = data[data['timestamp_[iso]'] >= start]
        if end:
            data = data[data['timestamp_[iso]'] <= end]
        return data

    def _create_connection(self):
        try:
            connection = sqlite3.connect(self._sql_db_path)
            print("Connection to SQLite DB successful")
            return connection
        except sqlite3.Error as e:
            raise ConnectionError(f"The error '{e}' occurred")

    def _execute_read_query(self, query):
        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")
            return []

    def _extract_table_list(self) -> list:
        read_query = """
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name;
            """
        table_list = [i[0] for i in self._execute_read_query(read_query)]
        return table_list

    def _check_and_convert_datetime_string(self, a_string: str):
        try:
            a_string = datetime.fromisoformat(a_string)
        except:
            raise ValueError(f'The string \'{a_string}\' is not in iso format')
            # If the input string is not in ISO format, raise a ValueError exception with an informative message.
            # The exception will be caught by the caller of this function, allowing them to handle the error.


    def _separate_data_unit(self, df: pd.DataFrame):
        unit_dict = {}

        for c in df.columns:
            n, u = c.split('_[')
            u = u[:-1]
            unit_dict[n] = u
            df.rename(columns={c: n}, inplace=True)

        return df, unit_dict
    
    def drop_and_create_dataframe(df, columns_to_drop):
        return df.drop(columns=columns_to_drop, axis=1)
    
    def load_data(db_path: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        reader = SQLDataReader(db_path)
        data, unit = reader.get_data_pandas(start=start_time, end=end_time)
        return data

    def prepare_data(df: pd.DataFrame, columns_to_keep: list) -> pd.DataFrame:
        df = df.dropna()
        columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
        return df.drop(columns=columns_to_drop)
    
    def preprocesing(lab_data,mvx_data):
        conc = lab_data.merge(mvx_data)
        null_rows = conc.isnull().any(axis=1).sum()
        merged_df = conc.dropna()
        columns_to_drop = [col for col in merged_df.columns if 'std' in col]
        merged_df.drop(columns=columns_to_drop, inplace=True)
        merged_df.set_index('timestamp', inplace=True)
        # Creating the separate dataframes using a dictionary and a loop
        df_names = ['df_sulphur', 'df_turb', 'df_nh', 'df_po4', 'df_doc', 'df_nsol']
        drop_columns_dict = {
            'df_sulphur': ['doc', 'po4', 'nh4', 'nsol', 'turbidity'],
            'df_turb': ['doc', 'po4', 'nh4', 'nsol', 'so4'],
            'df_nh': ['doc', 'po4', 'turbidity', 'nsol', 'so4'],
            'df_po4': ['doc', 'nh4', 'turbidity', 'nsol', 'so4'],
            'df_doc': ['po4', 'nh4', 'turbidity', 'nsol', 'so4'],
            'df_nsol': ['po4', 'nh4', 'turbidity', 'doc', 'so4']
        }

        dataframes = {}
        for df_name in df_names:
            dataframes[df_name] = SQLDataReader.drop_and_create_dataframe(merged_df, drop_columns_dict[df_name])
        return dataframes