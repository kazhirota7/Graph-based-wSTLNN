# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("state_covid_data/us-states.csv.txt")
df = df[['date', 'state', 'fips', 'deaths', 'cases']]
arizona = df.loc[df['state'] == 'Arizona']
california = df.loc[df['state'] == 'California']
nevada = df.loc[df['state'] == 'Nevada']
new_mexico = df.loc[df['state'] == 'New Mexico']
utah = df.loc[df['state'] == 'Utah']

arizona.to_csv(r'C:\Users\Kaz Hirota\Desktop\wSTLNN\wSTLNN\state_covid_data\arizona.txt', header=["date", "state", "fips", "deaths", "cases"], index=None, mode='w')
california.to_csv(r'C:\Users\Kaz Hirota\Desktop\wSTLNN\wSTLNN\state_covid_data\california.txt', header=["date", "state", "fips", "deaths", "cases"], index=None, mode='w')
nevada.to_csv(r'C:\Users\Kaz Hirota\Desktop\wSTLNN\wSTLNN\state_covid_data\nevada.txt', header=["date", "state", "fips", "deaths", "cases"], index=None, mode='w')
new_mexico.to_csv(r'C:\Users\Kaz Hirota\Desktop\wSTLNN\wSTLNN\state_covid_data\new_mexico.txt', header=["date", "state", "fips", "deaths", "cases"], index=None, mode='w')
utah.to_csv(r'C:\Users\Kaz Hirota\Desktop\wSTLNN\wSTLNN\state_covid_data\utah.txt', header=["date", "state", "fips", "deaths", "cases"], index=None, mode='w')


