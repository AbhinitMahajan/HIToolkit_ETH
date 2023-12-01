
![Screenshot 2023-11-09 182356](https://github.com/AbhinitMahajan/HIToolkit_ETH/assets/82913786/994bb1c2-255e-4e27-b7d9-4c91e83621f4)

---

# Non-contact Water Quality Monitoring using ML and Hyperspectral Imaging

## Description
A toolkit that offers users the ability to implement various preprocessing techniques on hyperspectral images of flowing wastewater. It also assists users in implementing and fine-tuning multiple machine learning models.

## Installation
To install the `HI_package`, use the following pip command:
```bash
pip install git+https://github.com/AbhinitMahajan/HIToolkit_ETH.git
```

## Usage

#### Loading Data
Ensure you provide the correct path and timestamp to load the data 


```python
from HI_package.your_sql_reader_module import SQLDataReader
### Load data
# Please make sure you put in the path and timestamp correctly
mvx_data = SQLDataReader.load_data('230817_sql/flume_mvx_spectra.sqlite')
scan_data = SQLDataReader.load_data('230817_sql/flume_scan_spectra.sqlite', '2023-05-08T09:00:00', '2023-09-01T00:00:00')
lab_data = SQLDataReader.load_data('230817_sql/lab_data.sqlite')
lab_data.drop(columns=['toc', 'ntot', 'tss'], inplace=True)  
turbimax_data = SQLDataReader.load_data('230817_sql/flume_turbimax_data.sqlite', '2023-05-08T09:00:00', '2023-09-01T00:00:00')


### Producing dataframes  OPTIONAL ONLY
# Preprocessing function inside class SQLDataReader
# The following code can be used to retrive dataframes: df_sulphur, df_turb, df_nh, df_po4, df_doc, df_nsol
# Produce a dictionary of dataframes which you can access 
# SQLDataReader.preprocesing takes 2 inputs: measurements(lab_data) and spectra(mvx_data(reflectance) or scan_data(absorbance)) 
dataframes = SQLDataReader.preprocesing(lab_data, mvx_data)
```

#### Modelling Data
Just run the command main_script.main to start the modeling process as shown 
- Takes two inputs - dataframe and target_column
- Example df_sulphur & 'sulphur'
```python
from HI_package import your_sql_reader_module 
from HI_package import imports_module 
from HI_package import functions_module 
from HI_package import main_script

main_script.main(df, target_column)
```

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Credits/Acknowledgments
A special shoutout to PhD student @ETH, Pierre Lechevalier, for his invaluable contributions and insights.

---

