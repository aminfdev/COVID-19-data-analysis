# COVID-19 Data Analysis
# Written by Mohammad Amin Fathi & Mehdi Afshari
# Institute for Advanced Studies in Basic Sciences (IASBS)
# Department of Computer Science and Information Technology

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from continent import country_to_continent

data_confirmed = pd.read_csv("./data-sets/time_series_covid19_confirmed_global.csv")
data_deaths = pd.read_csv("./data-sets/time_series_covid19_deaths_global.csv")
data_recovered = pd.read_csv("./data-sets/time_series_covid19_recovered_global.csv")
data_countries = pd.read_csv("./data-sets/UID_ISO_FIPS_LookUp_Table.csv")

start_date = "1/22/20"
end_date = "12/13/20"


# //////////////////////////////////////////////////////////////////////////
# Creating the Datas and Pre-Proccessing Them Befor Generating the Main Data
# //////////////////////////////////////////////////////////////////////////

data_confirmed = data_confirmed.groupby(
    'Country/Region').sum(numeric_only=True).reset_index()
data_deaths = data_deaths.groupby(
    'Country/Region').sum(numeric_only=True).reset_index()
data_recovered = data_recovered.groupby(
    'Country/Region').sum(numeric_only=True).reset_index()
data_population = data_countries.groupby(
    'Country_Region').sum(numeric_only=False).reset_index()
data_countries = data_countries.groupby(
    'Country_Region').sum(numeric_only=False).reset_index()

data_iso = data_countries.drop_duplicates(subset=["iso3"])

countries_to_drop = ["Diamond Princess", "MS Zaandam",
                     "Summer Olympics 2020", "Winter Olympics 2022", "Antarctica"]
iso_to_drop = ["ESH", "ATA"]

data_confirmed = data_confirmed[~data_confirmed["Country/Region"].isin(
    countries_to_drop)].reset_index()
data_deaths = data_deaths[~data_deaths["Country/Region"]
                          .isin(countries_to_drop)].reset_index()
data_recovered = data_recovered[~data_recovered["Country/Region"].isin(
    countries_to_drop)].reset_index()
data_population = data_population[~data_population["iso3"].isin(
    iso_to_drop)].reset_index()
data_population = data_population[~data_population["Country_Region"].isin(
    countries_to_drop)].reset_index()
data_iso = data_iso[~data_iso["Country_Region"].isin(countries_to_drop)]
data_iso = data_iso.dropna(subset=["iso3"]).reset_index()
data_iso = data_iso[~data_iso["iso3"].isin(iso_to_drop)].reset_index()


# /////////////////////////////
# Data Frame Generator Function
# /////////////////////////////

def create_data_frame(date):

    countries = data_confirmed["Country/Region"]
    iso3 = data_iso["iso3"].str[:3]
    lat = data_confirmed["Lat"]
    long = data_confirmed["Long"]
    confirmed = data_confirmed[date]
    deaths = data_deaths[date]
    recovered = data_recovered[date]
    population = data_population["Population"]

    data_frame = pd.DataFrame({
        "ISO-3166": iso3,
        'Country': countries,
        "Last Update": date,
        "Lat": lat,
        "Long": long,
        'Confirmed': confirmed,
        "Deaths": deaths,
        "Recovered": recovered,
        "Active": (confirmed - (deaths + recovered)),
        "Incident_Rate": ((confirmed / population) * 1000000),
        "Mortality_Rate (per 100)": ((deaths / population) * 100)
    })

    data_frame['Continent'] = data_frame['Country'].map(country_to_continent)
    data_frame.insert(2, 'Continent', data_frame.pop('Continent'))

    data_frame = data_frame.loc[(data_frame[["Confirmed"]] != 0).any(axis=1)]
    data_frame = data_frame.reset_index(drop=True)

    return data_frame


# /////////////////////////////////////////////////
# Generating the Main Data and Save it to the Files
# /////////////////////////////////////////////////

data = create_data_frame(end_date)

data.to_csv("./data-sets/data.csv", index=False)


# //////////////////////////////////////////////////
# General Analysis of Data - Table 1 - Global Report
# //////////////////////////////////////////////////

data_sum = data.sum(numeric_only=True)
data_sum = data_sum.to_frame().transpose()
data_sum = data_sum.drop(["Lat", "Long"], axis=1)

styled_data = data_sum.style.background_gradient(cmap="Wistia", axis=1)


# //////////////////////////////////////////////////////////
# General Analysis of Data - Table 2 - Continent Wise Report
# //////////////////////////////////////////////////////////

data_continent_wise = data.drop(
    ["ISO-3166", "Country", "Last Update", "Lat", "Long"], axis=1)
data_continent_wise = data_continent_wise.groupby(
    'Continent').sum(numeric_only=True)
data_continent_wise = data_continent_wise.sort_values(
    by='Confirmed', ascending=False).reset_index()

styled_data = data_continent_wise.style\
    .background_gradient(cmap='Blues', subset=["Confirmed"])\
    .background_gradient(cmap='Reds', subset=["Deaths"])\
    .background_gradient(cmap='Greens', subset=["Recovered"])\
    .background_gradient(cmap='Purples', subset=["Active"])\
    .background_gradient(cmap='GnBu', subset=["Incident_Rate"])\
    .background_gradient(cmap='OrRd', subset=["Mortality_Rate (per 100)"])


# ////////////////////////////////////////////////////////
# General Analysis of Data - Table 3 - Country Wise Report
# ////////////////////////////////////////////////////////

data_sorted = data.sort_values(by='Confirmed', ascending=False)
data_sorted = data_sorted.drop(
    ["ISO-3166", "Continent", "Last Update", "Lat", "Long"], axis=1).reset_index(drop=True)

styled_data = data_sorted.head(15).style\
    .background_gradient(cmap='Blues', subset=["Confirmed"])\
    .background_gradient(cmap='Reds', subset=["Deaths"])\
    .background_gradient(cmap='Greens', subset=["Recovered"])\
    .background_gradient(cmap='Purples', subset=["Active"])\
    .background_gradient(cmap='GnBu', subset=["Incident_Rate"])\
    .background_gradient(cmap='OrRd', subset=["Mortality_Rate (per 100)"])


# /////////////////////////////////////////////////////////////////
# General Analysis of Data - Visualizing Plots for Top 10 Countries
# /////////////////////////////////////////////////////////////////

top_10_confirmed = data.nlargest(10, 'Confirmed')
top_10_confirmed = top_10_confirmed.sort_values(by='Confirmed', ascending=True)

top_10_deaths = data.nlargest(10, 'Deaths')
top_10_deaths = top_10_deaths.sort_values(by='Deaths', ascending=True)

top_10_recovered = data.nlargest(10, 'Recovered')
top_10_recovered = top_10_recovered.sort_values(by='Recovered', ascending=True)

top_10_active = data.nlargest(10, 'Active')
top_10_active = top_10_active.sort_values(by='Active', ascending=True)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

axes[0, 0].barh(top_10_confirmed['Country'], top_10_confirmed['Confirmed'])
axes[0, 0].set_title('Top 10 Countries (Confirmed Cases)')

axes[0, 1].barh(top_10_deaths['Country'], top_10_deaths['Deaths'], color='red')
axes[0, 1].set_title('Top 10 Countries (Deaths)')

axes[1, 0].barh(top_10_recovered['Country'],
                top_10_recovered['Recovered'], color='green')
axes[1, 0].set_title('Top 10 Countries (Recovered Cases)')

axes[1, 1].barh(top_10_active['Country'],
                top_10_active['Active'], color='orange')
axes[1, 1].set_title('Top 10 Countries (Active Cases)')

plt.tight_layout()
plt.show()


# //////////////////////////////////////////
# Correlation Analysis - Country Wise Report
# //////////////////////////////////////////

correlation_matrix = data[['Confirmed', 'Deaths', 'Recovered', 'Active',
                           "Incident_Rate", "Mortality_Rate (per 100)"]].corr(method='pearson')

styled_data = correlation_matrix.style\
    .background_gradient(cmap='OrRd', subset=["Confirmed"])\
    .background_gradient(cmap='OrRd', subset=["Deaths"])\
    .background_gradient(cmap='OrRd', subset=["Recovered"])\
    .background_gradient(cmap='OrRd', subset=["Active"])\
    .background_gradient(cmap='OrRd', subset=["Incident_Rate"])\
    .background_gradient(cmap='OrRd', subset=["Mortality_Rate (per 100)"])


# ////////////////////////////////////////////
# Correlation Analysis - Continent Wise Report
# ////////////////////////////////////////////

data_continent_wise = data.groupby('Continent').sum(numeric_only=True)
correlation_matrix = data_continent_wise[['Confirmed', 'Deaths', 'Recovered',
                                          'Active', "Incident_Rate", "Mortality_Rate (per 100)"]].corr(method='pearson')

styled_data = correlation_matrix.style\
    .background_gradient(cmap='OrRd', subset=["Confirmed"])\
    .background_gradient(cmap='OrRd', subset=["Deaths"])\
    .background_gradient(cmap='OrRd', subset=["Recovered"])\
    .background_gradient(cmap='OrRd', subset=["Active"])\
    .background_gradient(cmap='OrRd', subset=["Incident_Rate"])\
    .background_gradient(cmap='OrRd', subset=["Mortality_Rate (per 100)"])


# ////////////////////////////////////////////////////////////
# Spread Analysis - Number of Affected Countries over the Time
# ////////////////////////////////////////////////////////////

def create_daily_data_frame(start_date, end_date, target_column, index_name):

    dates_list = data_confirmed.loc[:, start_date:end_date]
    dates_list = dates_list.columns.tolist()

    data_frame = pd.DataFrame(index=[index_name])

    non_zero_counts = []

    for date in dates_list:
        df = create_data_frame(date)
        count = np.count_nonzero(df[target_column])
        non_zero_counts.append(count)

    data_frame = pd.DataFrame(
        [non_zero_counts],
        index=[index_name],
        columns=dates_list
    )

    return data_frame


# ////////////////////////////////////////////////////////////
# Spread Analysis - Number of Affected Countries over the Time
# ////////////////////////////////////////////////////////////

affected_countries_count = create_daily_data_frame(
    start_date, end_date, "Confirmed", 'Affected Countries Count')

plt.figure(figsize=(20, 10))
plt.plot(affected_countries_count.columns,
         affected_countries_count.values[0], marker='o', linestyle='-')

plt.title('Affected Countries Over Time')
plt.xlabel(f'Days ({start_date} - {end_date})')
plt.ylabel('Number of Affected Countries')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ////////////////////////////////////
# Spread Analysis - Global Daily Cases
# ////////////////////////////////////

def create_daily_data_frame_global(start_date, end_date, target_column, index_name):

    dates_list = data_confirmed.loc[:, start_date:end_date]
    dates_list = dates_list.columns.tolist()

    data_frame = pd.DataFrame(index=[index_name])

    counts = []

    for date in dates_list:
        df = create_data_frame(date)
        count = df[target_column].sum()
        counts.append(count)

    data_frame = pd.DataFrame(
        [counts],
        index=[index_name],
        columns=dates_list
    )

    return data_frame


# ////////////////////////////////////
# Spread Analysis - Global Daily Cases
# ////////////////////////////////////

global_confirmed = create_daily_data_frame_global(
    start_date, end_date, "Confirmed", "Global")
global_deaths = create_daily_data_frame_global(
    start_date, end_date, "Deaths", "Global")
global_Recovered = create_daily_data_frame_global(
    start_date, end_date, "Recovered", "Global")
global_Active = create_daily_data_frame_global(
    start_date, end_date, "Active", "Global")

plt.figure(figsize=(20, 12))
plt.plot(global_confirmed.columns,
         global_confirmed.values[0], marker='o', linestyle='-', color='skyblue', label="Confirmed")
plt.plot(global_deaths.columns,
         global_deaths.values[0], marker='o', linestyle='-', color='red', label="Deaths")
plt.plot(global_Recovered.columns,
         global_Recovered.values[0], marker='o', linestyle='-', color='green', label="Recovered")
plt.plot(global_Active.columns,
         global_Active.values[0], marker='o', linestyle='-', color='orange', label="Active")

plt.fill_between(global_confirmed.columns,
                 global_confirmed.values[0], linestyle='-', color='skyblue', alpha=0.3)
plt.fill_between(global_deaths.columns,
                 global_deaths.values[0], linestyle='-', color='red', alpha=0.3)
plt.fill_between(global_Recovered.columns,
                 global_Recovered.values[0], linestyle='-', color='green', alpha=0.3)
plt.fill_between(global_Active.columns,
                 global_Active.values[0], linestyle='-', color='orange', alpha=0.3)

plt.title('Daily Cases - Global')
plt.xlabel(f'Days ({start_date} - {end_date})')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ////////////////////////////////////////////
# Spread Analysis - Continent Wise Daily Cases
# ////////////////////////////////////////////

def create_daily_data_frame_for_continent(start_date, end_date, target_column, continent_name):

    dates_list = data_confirmed.loc[:, start_date:end_date]
    dates_list = dates_list.columns.tolist()

    data_frame = pd.DataFrame(index=[continent_name])

    counts = []

    for date in dates_list:
        df = create_data_frame(date)
        df = df.groupby('Continent').sum(numeric_only=True)

        if continent_name in df.index:
            count = df.loc[continent_name, target_column]
        else:
            count = 0

        counts.append(count)

    data_frame = pd.DataFrame(
        [counts],
        index=[continent_name],
        columns=dates_list
    )

    return data_frame


# ////////////////////////////////////////////
# Spread Analysis - Continent Wise Daily Cases
# ////////////////////////////////////////////

continents = ["Asia", "Africa", "Europe",
              "North America", "South America", "Oceania"]
num_countries = len(continents)
num_plots_per_row = 2
num_rows = num_countries // num_plots_per_row if num_countries % num_plots_per_row == 0 else num_countries // num_plots_per_row + 1

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 15))

for index, continent in enumerate(continents):
    confirmed_data_daily = create_daily_data_frame_for_continent(
        start_date, end_date, "Confirmed", continent)
    death_data_daily = create_daily_data_frame_for_continent(
        start_date, end_date, "Deaths", continent)
    recovered_data_daily = create_daily_data_frame_for_continent(
        start_date, end_date, "Recovered", continent)
    active_data_daily = create_daily_data_frame_for_continent(
        start_date, end_date, "Active", continent)

    row = index // num_plots_per_row
    col = index % num_plots_per_row

    axes[row, col].plot(confirmed_data_daily.columns, confirmed_data_daily.values[0],
                        marker='o', linestyle='-', color='skyblue', label='Confirmed')
    axes[row, col].plot(death_data_daily.columns, death_data_daily.values[0],
                        marker='o', linestyle='-', color='red', label='Deaths')
    axes[row, col].plot(recovered_data_daily.columns, recovered_data_daily.values[0],
                        marker='o', linestyle='-', color='green', label='Recovered')
    axes[row, col].plot(active_data_daily.columns, active_data_daily.values[0],
                        marker='o', linestyle='-', color='orange', label='Active')

    axes[row, col].fill_between(confirmed_data_daily.columns,
                                confirmed_data_daily.values[0], color='skyblue', alpha=0.3)
    axes[row, col].fill_between(
        death_data_daily.columns, death_data_daily.values[0], color='red', alpha=0.3)
    axes[row, col].fill_between(recovered_data_daily.columns,
                                recovered_data_daily.values[0], color='green', alpha=0.3)
    axes[row, col].fill_between(
        active_data_daily.columns, active_data_daily.values[0], color='orange', alpha=0.3)

    axes[row, col].set_title(f'Daily Cases in {continent}')
    axes[row, col].set_xlabel(f'Days ({start_date} - {end_date})')
    axes[row, col].set_ylabel('Number of Cases')
    axes[row, col].legend()
    axes[row, col].grid(True)

plt.tight_layout()
plt.show()


# //////////////////////////////////////////
# Spread Analysis - Country Wise Daily Cases
# //////////////////////////////////////////

def create_daily_data_frame_for_country(start_date, end_date, target_column, country_name):

    dates_list = data_confirmed.loc[:, start_date:end_date]
    dates_list = dates_list.columns.tolist()

    data_frame = pd.DataFrame(index=[country_name])

    counts = []

    for date in dates_list:
        df = create_data_frame(date)
        count = 0
        if country_name in df['Country'].values:
            count = df[df['Country'] == country_name][target_column].values[0]
        counts.append(count)

    data_frame = pd.DataFrame(
        [counts],
        index=[country_name],
        columns=dates_list
    )

    return data_frame


# //////////////////////////////////////////
# Spread Analysis - Country Wise Daily Cases
# //////////////////////////////////////////

countries = ["US", "Russia", "India", "Iran"]
num_countries = len(countries)
num_plots_per_row = 2
num_rows = num_countries // num_plots_per_row if num_countries % num_plots_per_row == 0 else num_countries // num_plots_per_row + 1

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 10))

for index, country in enumerate(countries):
    confirmed_data_daily = create_daily_data_frame_for_country(
        start_date, end_date, "Confirmed", country)
    death_data_daily = create_daily_data_frame_for_country(
        start_date, end_date, "Deaths", country)
    recovered_data_daily = create_daily_data_frame_for_country(
        start_date, end_date, "Recovered", country)
    active_data_daily = create_daily_data_frame_for_country(
        start_date, end_date, "Active", country)

    row = index // num_plots_per_row
    col = index % num_plots_per_row

    axes[row, col].plot(confirmed_data_daily.columns, confirmed_data_daily.values[0],
                        marker='o', linestyle='-', color='skyblue', label='Confirmed')
    axes[row, col].plot(death_data_daily.columns, death_data_daily.values[0],
                        marker='o', linestyle='-', color='red', label='Deaths')
    axes[row, col].plot(recovered_data_daily.columns, recovered_data_daily.values[0],
                        marker='o', linestyle='-', color='green', label='Recovered')
    axes[row, col].plot(active_data_daily.columns, active_data_daily.values[0],
                        marker='o', linestyle='-', color='orange', label='Active')

    axes[row, col].fill_between(confirmed_data_daily.columns,
                                confirmed_data_daily.values[0], color='skyblue', alpha=0.3)
    axes[row, col].fill_between(
        death_data_daily.columns, death_data_daily.values[0], color='red', alpha=0.3)
    axes[row, col].fill_between(recovered_data_daily.columns,
                                recovered_data_daily.values[0], color='green', alpha=0.3)
    axes[row, col].fill_between(
        active_data_daily.columns, active_data_daily.values[0], color='orange', alpha=0.3)

    axes[row, col].set_title(f'Daily Cases in {country}')
    axes[row, col].set_xlabel(f'Days ({start_date} - {end_date})')
    axes[row, col].set_ylabel('Number of Cases')
    axes[row, col].legend()
    axes[row, col].grid(True)

plt.tight_layout()
plt.show()


# /////////////////////////////////////////////////////////
# Spread Analysis - Trend Comparison of Different Countries
# /////////////////////////////////////////////////////////

countries = ["US", "India", "Brazil", "Russia", "France",
             "United Kingdom", "Italy", "Turkey", "Spain", "Argentina"]

plt.figure(figsize=(25, 10))

for country in countries:
    confirmed_data_daily = create_daily_data_frame_for_country(
        start_date, end_date, "Confirmed", country)
    plt.plot(confirmed_data_daily.columns,
             confirmed_data_daily.values[0], marker='o', linestyle='-', label=country)

plt.title('Daily Confirmed Cases in Different Countries')
plt.xlabel(f'Days ({start_date} - {end_date})')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.show()


# ////////////////////////////////////////////////////////////
# Spread Analysis - Number of New Cases in Different Countries
# ////////////////////////////////////////////////////////////

def create_daily_data_for_country(start_date, end_date, target_column, country_name):

    dates_list = data_confirmed.loc[:, start_date:end_date]
    dates_list = dates_list.columns.tolist()

    data_frame = pd.DataFrame(index=[country_name])

    counts = []

    for date in dates_list:
        df = create_data_frame(date)
        count = 0
        if country_name in df['Country'].values:
            count = df[df['Country'] == country_name][target_column].values[0]
        counts.append(count)

    counts.reverse()

    for i in range(0, len(counts) - 1):
        counts[i] = counts[i] - counts[i + 1]

    counts.reverse()

    data_frame = pd.DataFrame(
        [counts],
        index=[country_name],
        columns=dates_list
    )

    return data_frame


# ////////////////////////////////////////////////////////////
# Spread Analysis - Number of New Cases in Different Countries
# ////////////////////////////////////////////////////////////

countries = ["US", "India", "Germany", "Russia",
             "United Kingdom", "Italy", "Japan", "Argentina", "Iran"]
plt.figure(figsize=(25, 10))

for country in countries:
    confirmed_data_daily = create_daily_data_for_country(
        start_date, end_date, "Confirmed", country)
    plt.plot(confirmed_data_daily.columns,
             confirmed_data_daily.values[0], marker='o', linestyle='-', label=country)

plt.title('Daily Confirmed Cases in Different Countries')
plt.xlabel(f'Days ({start_date} - {end_date})')
plt.ylabel('Number of Cases')
plt.legend()
plt.grid(True)
plt.show()


# ///////////////////////////////////////////
# Visualizing Pie Plots for World Total Cases
# ///////////////////////////////////////////

def create_pie_plot_data_frame(data_frame, target_column, threshold):

    countries = data_frame[data_frame[target_column] > threshold]
    others = data_frame[data_frame[target_column] <= threshold]

    pie_data = pd.DataFrame({
        'Country': countries['Country'],
        'Value': countries[target_column]
    })

    others_sum = others[target_column].sum()

    others_data = pd.DataFrame({'Country': ['Others'],
                                'Value': [others_sum]})

    pie_data = pd.concat([pie_data, others_data], ignore_index=True)

    return pie_data


# ///////////////////////////////////////////
# Visualizing Pie Plots for World Total Cases
# ///////////////////////////////////////////

columns = [("Confirmed", 1000000), ("Deaths", 30000),
           ("Active", 200000), ("Recovered", 500000)]
num_columns = len(columns)
num_plots_per_row = 2
num_rows = num_columns // num_plots_per_row if num_columns % num_plots_per_row == 0 else num_columns // num_plots_per_row + 1

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 15))

for index, column in enumerate(columns):

    pie_data = create_pie_plot_data_frame(data, column[0], column[1])

    row = index // num_plots_per_row
    col = index % num_plots_per_row

    axes[row, col].pie(
        pie_data['Value'], labels=pie_data['Country'], autopct='%1.1f%%', startangle=90)
    axes[row, col].set_title(f'{column[0]} Cases Distribution')

plt.tight_layout()
plt.show()
