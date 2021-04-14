## §1. Create a Database

First, let's import important packages.


```python
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
import numpy as np
```


```python
import sqlite3
```


```python
conn = sqlite3.connect("temps.db") # create a database in current directory called temps.db
```

We will write a function to clean our data before incorporating it into our database.


```python
def prepare_df(df):
    '''
    this function takes a dataframe with months as columns names with temperature as values
    and returns a dataframe with month and temperature as column names 
    '''
    # convert all the columns that we don't want to stack into a multi-index for the data frame
    df = df.set_index(keys=["ID", "Year"])
    # stacking
    df = df.stack()
    # recover ID and Year columns
    df = df.reset_index()
    # rename columns to make them more readable
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    # extract the integer part to be month value
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    # this will be used later for joining
    df["FIPS 10-4"] = df["ID"].str[0:2]
    return(df)
```

Now, let's add data to our database!


```python
# temperature table 
df_iter = pd.read_csv("temps.csv", chunksize = 100000)
for df in df_iter:
    df = prepare_df(df)
    df.to_sql("temperatures", conn, if_exists = "append", index = False)
```

    C:\Users\35132\anaconda3\envs\PIC16B\lib\site-packages\pandas\core\generic.py:2779: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      sql.to_sql(
    


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
      <th>FIPS 10-4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>1</td>
      <td>-13.69</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>2</td>
      <td>-8.40</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>3</td>
      <td>-0.20</td>
      <td>US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>4</td>
      <td>3.21</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>5</td>
      <td>13.85</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>




```python
# stations table
station_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(station_url)
stations.to_sql("stations", conn, if_exists = "replace", index = False)
```


```python
stations.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>




```python
# countries table 
country_url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(country_url)
countries.to_sql("countries", conn, if_exists = "replace", index = False)
```

    C:\Users\35132\anaconda3\envs\PIC16B\lib\site-packages\pandas\core\generic.py:2779: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      sql.to_sql(
    


```python
countries.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS 10-4</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>



Let's check if we've correctly added data to our database with `cursor`.


```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
```

    [('temperatures',), ('stations',), ('countries',)]
    

Great! We've correctly added three tables to our database. It's also a good idea to check what's in each table.


```python
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "temperatures" (
    "ID" TEXT,
      "Year" INTEGER,
      "Month" INTEGER,
      "Temp" REAL,
      "FIPS 10-4" TEXT
    )
    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT
    )
    CREATE TABLE "countries" (
    "FIPS 10-4" TEXT,
      "ISO 3166" TEXT,
      "Name" TEXT
    )
    


```python
# close the database connection
conn.close()
```

## §2. Write a Query Function


```python
conn = sqlite3.connect("temps.db")
```

The SQL syntax is based on a similar idea as `pandas`.

- `SELECT`, like the syntax `[]`, controls which column(s) will be returned. 
- `FROM` tells us which table to return columns from. 
- `WHERE` is like the Boolean index `[temperatures["year"] == 1990]`. Only rows in which this criterion is satisfied will be returned. 


```python
def query_climate_database(country, year_begin, year_end, month):
    '''
    this function gives the temperature of a country within specified year range in the specific month.
    inputs: 
    country, a string giving the name of a country for which data should be returned.
    year_begin and year_end, two integers giving the earliest and latest years for which should be returned.
    month, an integer giving the month of the year for which should be returned.
    ouput:
    a Pandas dataframe of temperature readings for a country within specified year range in the specific month
    '''
    
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON T.'fips 10-4'= C.'fips 10-4'
    WHERE T.year BETWEEN ? AND ?
    AND T.month = ?
    AND C.name = ?
    """
    
    df = pd.read_sql_query(cmd, conn, params = (year_begin, year_end, month, country))
    # rename the column of country name to Country
    df.rename(columns={"Name": "Country"},inplace = True)
    return(df)
```


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12603</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>12604</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>12605</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>12606</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>12607</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>12608 rows × 7 columns</p>
</div>



## §3. Write a Geographic Scatter Function for Yearly Temperature Increases

In this section, we will use package `plotly` from module `plotly` to create interative visualizations! We want to explore how the average yearly change in temperature vary within a given country.


```python
from plotly import express as px
import calendar #convert number to month name 
```

To quantify average yearly change, we will compute the first coefficient of a linear regression model at that station.


```python
from sklearn.linear_model import LinearRegression

def coef(data_group):
    '''
    this function computes the first coefficient of the linear model Year vs. Temp
    '''
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
```

Now, we might want to have different colors for climate stations according to average yearly change in temperature, which can be computed using the function we just wrote above. A good idea is to use geographic scatterplot. 


```python
def temperature_coefficient_plot(country,year_begin, year_end, month, min_obs, **kwargs):
    '''
    this functions generates an interactive geographic scatterplot to visualize yearly average change in temperature 
    in a specific month in a given country during specified years.
    Inputs:
    country, a string giving the name of a country for which data should be returned.
    year_begin and year_end, two integers giving the earliest and latest years for which should be returned.
    month, an integer giving the month of the year for which should be returned.
    min_obs:　the minimum required number of years of data for any given station.
    output:
    a geographic scatterplot
    '''
    
    # create a dataframe containing all the information needed
    df = query_climate_database(country, year_begin, year_end, month)
    # count the number of years of data for each temperature station
    df['freq'] = df.groupby('NAME')['NAME'].transform('count')
    # only keep data for stations with at least min_obs years worth of data  
    df = df[df['freq']>= min_obs]
    c = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef).reset_index()    
    # add a new column to store coefficients round up to 4 decimal places
    z = "Estimated Yearly Increase " + u"(\N{DEGREE SIGN}C)"
    c[z] = c[0].round(decimals = 4)
    
    title = "Estimte of yearly increase in temperature in "+ list(calendar.month_name)[month] +" <br>for stations in "+ country +" , years "+ str(year_begin)+" - "+str(year_end)
    fig = px.scatter_mapbox(c, 
                        lat = "LATITUDE",
                        lon = "LONGITUDE", 
                        hover_name = "NAME",
                        color = z,
                        title = title,
                        **kwargs)
    return(fig)
```


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig1 = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_midpoint = 0, # 0 corresponds to the center of colorbar
                                   color_continuous_scale=color_map)

fig1.show()
```
{% include geo_scatter.html %}




```python
from plotly.io import write_html
write_html(fig1, "geo_scatter.html")
```

The colors get brighter when the temperature increases more. To learn about the information for a temperature station, we can simply put our cursor on the dot representing that station, then we will be able to read its latitude, longitude, and estimated yearly increase in Celcius.

## §4. Create Two More Interesting Figures

Is there a relationship between the elevation of a temperature station and its yearly increase in temperature? To explore this question, it's a good idea to control variables. To be specific, we want to control latitude and month (we assume temperature change is not closely related to longitude). As we did above, we will first write a query function to create the dataframe desired.


```python
def elevation_query(latitude_min, latitude_max, month):
    '''
    this function gives the temperature of countries within specified latitude range in the specific month.
    inputs: 
    latitude_min and latitude_max, two doubles giving the smallest and largest latitude.
    month, an integer giving the month of the year for which should be returned.
    ouput:
    a Pandas dataframe of temperature readings for countrys within specified latitude range in the specific month
    '''
    
    cmd = \
    """
    SELECT S.name, T.year, T.month, T.temp, S.STNELEV, C.name
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON T.'fips 10-4'= C.'fips 10-4'
    WHERE S.latitude BETWEEN ? AND ?
    AND T.month = ?
    """

    df = pd.read_sql_query(cmd, conn, params = (latitude_min, latitude_max, month))
    # rename the column containing country names to avoid confusion
    df.rename(columns={"Name": "Country"},inplace = True)
    return(df)
```

scatterplot can help us visualize correlations easily.We want to examine whether there's an correlation between elevation and yearly change in temperature within each country when latitude and month is constant. 


```python
def elevation_temp_scatter(latitude_min, latitude_max, month, min_obs, **kwargs):
    '''
    this functions generates an interactive scatterplot to visualize yearly average change in temperature v.s.elevation
    in a specific month for countries within specified latitude range.
    Inputs:
    latitude_min and latitude_max, two doubles giving the smallest and largest latitude.
    month, an integer giving the month of the year for which should be returned.
    min_obs:　the minimum required number of stations for any given country.
    output:
    a scatterplot
    '''
    
    # create a dataframe containing all the information needed
    df = elevation_query(latitude_min, latitude_max, month)
    # obtain the yearly change in temeprature
    c = df.groupby(["NAME","STNELEV", "Country"]).apply(coef).reset_index()
    # rename columns for convenience
    z = "Yearly change in Temperatures " + u"(\N{DEGREE SIGN}C)"
    c.rename(columns = {"NAME": "Station", 0:z, "STNELEV": "Elevation"},inplace = True)
    # compute the number of different stations within each country and store the info as a new col
    c['freq'] = c.groupby('Country')['Country'].transform('count')
    # only keep data for countries with at least min_obs many stations 
    c = c[c['freq']>= min_obs] 
    # name for y_axis
    y_axis = "Yearly change in Temperatures " + u"(\N{DEGREE SIGN}C) between latitude "+ str(latitude_min)+" and "+str(latitude_max)+ " in "+ list(calendar.month_name)[month]

    fig = px.scatter(data_frame = c, 
                     x = "Elevation", 
                     y = z,
                     labels = {"Elevation" : "Elevation",
                               z : y_axis,
                               'Country' : 'Countries'},
                     hover_name = "Station",
                     hover_data = ["Elevation", z],
                     color = "Country",
                     **kwargs)
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return(fig)
```


```python
fig2 = elevation_temp_scatter(10,15,1, min_obs = 10)
fig2.show()
```
{% include scatter.html %}


```python
write_html(fig2, "scatter.html")
```

This graph could look overwhelming at the first sight, and it's hard to tell any patterns when all colors of dots are kind of just clustered. Don't worry! If you double click on the country you are interested in on the lengend, the plot will immediately give you only the dots that are under that country. For example, if we double click on Colombia, we can see that there is no obvious pattern between elevation and yearly change in temperature. This conclusion actually applies to almost all the countries for our selection. Therefore, further inquery is needed to figure out the relationship between elevation and yearly temperature change.

In the next visualization, I want to explore if there's difference in yearly change in temperature between two countries that are pretty close to each other.


```python
def country_query(country1, country2):
    '''
    this function gives the temperature of two chosen countries
    inputs: 
    country 1 and country2: two strings that gives the names of two different countries
    ouput:
    a Pandas dataframe of temperature readings for the two chosen countrys
    '''
    
    cmd = \
    """
    SELECT S.name, T.year, T.month, T.temp, C.name
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C ON T.'fips 10-4'= C.'fips 10-4'
    WHERE C.name IN (?,?)    
    """

    df = pd.read_sql_query(cmd, conn, params = (country1, country2))
    # rename the column containing country names to avoid confusion
    df.rename(columns={"Name": "Country"},inplace = True)
    return(df)
```

Boxplot presents a nice summary of our measurement of interest. With boxplot, we can easily compare the median, max, min value of the measurement between two countries.


```python
def country_comparison(country1, country2, **kwargs):
    '''
    this functions generates an interactive boxplot to visualize yearly average change in temperature 
    in two given countries for each month.
    Inputs:
    country 1 and country2: two strings that gives the names of two different countries
    output:
    a boxplot
    '''
    df = country_query(country1, country2)
    # yearly change in temperature over years for each month
    c = df.groupby(["NAME", "Month", "Country"]).apply(coef).reset_index()
    z = "Yearly change in Temperatures " + u"(\N{DEGREE SIGN}C)"
    c.rename(columns = {"NAME": "Station", 0: z},inplace = True)
    fig = px.box(c, 
                 x = "Month", 
                 y = z,
                 color = "Country")
    return(fig)
```


```python
fig3 = country_comparison("India", "Afghanistan")
```


```python
fig3.show()
```
{% include boxplot.html %}


```python
write_html(fig3, "boxplot.html")
```

From the plot, we can easily see that Idian's temperature is quite stable throughout the whole year for each year compared with Afghanistan. 


```python
conn.close()
```
