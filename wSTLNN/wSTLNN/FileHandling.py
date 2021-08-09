# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('AustWeather.csv', dtype={'Date':str, 'Location':str, 'MinTemp':float, 'MaxTemp':float, 'Rainfall':float, 'Evaporation':float, 'Sunshine':float, 'WindGustDir':str, 'WindGustSpeed':float, 'WindDir9am':str, 'WindDir3apm':str, 'WindSpeed9am':float, 'WindSpeed3pm':float, 'Pressure9am':float, 'Pressure3pm':float, 'Cloud9am':float, 'Cloud3pm':float, 'Temp9am':float, 'Temp3pm':float, 'RainToday':str, "RainTomorrow":str})

locations = df.groupby(df.Location)
albury = locations.get_group("Albury")
albury.to_excel("weather_data/albury.xlsx")

badgeryscreek = locations.get_group("BadgerysCreek")
badgeryscreek.to_excel("weather_data/badgeryscreek.xlsx")

cobar = locations.get_group("Cobar")
cobar.to_excel("weather_data/cobar.xlsx")

coffsharbour = locations.get_group("CoffsHarbour")
coffsharbour.to_excel("weather_data/coffsharbour.xlsx")

moree = locations.get_group("Moree")
moree.to_excel("weather_data/moree.xlsx")

newcastle = locations.get_group("Newcastle")
newcastle.to_excel("weather_data/newcastle.xlsx")

norahhead = locations.get_group("0rahHead")
norahhead.to_excel("weather_data/norahhead.xlsx")

norfolkisland = locations.get_group("0rfolkIsland")
norfolkisland.to_excel("weather_data/norfolkisland.xlsx")

penrith = locations.get_group("Penrith")
penrith.to_excel("weather_data/penrith.xlsx")

richmond = locations.get_group("Richmond")
richmond.to_excel("weather_data/richmond.xlsx")

sydney = locations.get_group("Sydney")
sydney.to_excel("weather_data/sydney.xlsx")

sydneyairport = locations.get_group("SydneyAirport")
sydneyairport.to_excel("weather_data/sydneyairport.xlsx")

waggawagga = locations.get_group("WaggaWagga")
waggawagga.to_excel("weather_data/waggawagga.xlsx")

williamtown = locations.get_group("Williamtown")
williamtown.to_excel("weather_data/williamtown.xlsx")

wollongong = locations.get_group("Wollongong")
wollongong.to_excel("weather_data/wollongong.xlsx")

canberra = locations.get_group("Canberra")
canberra.to_excel("weather_data/canberra.xlsx")

tuggeranong = locations.get_group("Tuggera0ng")
tuggeranong.to_excel("weather_data/tuggeranong.xlsx")

mountginini = locations.get_group("MountGinini")
mountginini.to_excel("weather_data/mountginini.xlsx")

ballarat = locations.get_group("Ballarat")
ballarat.to_excel("weather_data/ballarat.xlsx")

bendigo = locations.get_group("Bendigo")
bendigo.to_excel("weather_data/bendigo.xlsx")

sale = locations.get_group("Sale")
sale.to_excel("weather_data/sale.xlsx")

melbourne = locations.get_group("Melbourne")
melbourne.to_excel("weather_data/melbourne.xlsx")

melbourneairport = locations.get_group("MelbourneAirport")
melbourneairport.to_excel("weather_data/melbourneairport.xlsx")

mildura = locations.get_group("Mildura")
mildura.to_excel("weather_data/mildura.xlsx")

nhil = locations.get_group("Nhil")
nhil.to_excel("weather_data/nhil.xlsx")

portland = locations.get_group("Portland")
portland.to_excel("weather_data/portland.xlsx")

watsonia = locations.get_group("Watsonia")
watsonia.to_excel("weather_data/watsonia.xlsx")

dartmoor = locations.get_group("Dartmoor")
dartmoor.to_excel("weather_data/dartmoor.xlsx")

brisbane = locations.get_group("Brisbane")
brisbane.to_excel("weather_data/brisbane.xlsx")

cairns = locations.get_group("Cairns")
cairns.to_excel("weather_data/cairns.xlsx")

goldcoast = locations.get_group("GoldCoast")
goldcoast.to_excel("weather_data/goldcoast.xlsx")

townsville = locations.get_group("Townsville")
townsville.to_excel("weather_data/townsville.xlsx")

adelaide = locations.get_group("Adelaide")
adelaide.to_excel("weather_data/adelaide.xlsx")

mountgambier = locations.get_group("MountGambier")
mountgambier.to_excel("weather_data/mountgambier.xlsx")

nuriootpa = locations.get_group("Nuriootpa")
nuriootpa.to_excel("weather_data/nuriootpa.xlsx")

woomera = locations.get_group("Woomera")
woomera.to_excel("weather_data/woomera.xlsx")

albany = locations.get_group("Albany")
albany.to_excel("weather_data/albany.xlsx")

witchcliffe = locations.get_group("Witchcliffe")
witchcliffe.to_excel("weather_data/witchcliffe.xlsx")

pearceraaf = locations.get_group("PearceRAAF")
pearceraaf.to_excel("weather_data/pearceraaf.xlsx")

perthairport = locations.get_group("PerthAirport")
perthairport.to_excel("weather_data/perthairport.xlsx")

perth = locations.get_group("Perth")
perth.to_excel("weather_data/perth.xlsx")

salmongums = locations.get_group("SalmonGums")
salmongums.to_excel("weather_data/salmongums.xlsx")

walpole = locations.get_group("Walpole")
walpole.to_excel("weather_data/walpole.xlsx")

hobart = locations.get_group("Hobart")
hobart.to_excel("weather_data/hobart.xlsx")

launceston = locations.get_group("Launceston")
launceston.to_excel("weather_data/launceston.xlsx")

alicesprings = locations.get_group("AliceSprings")
alicesprings.to_excel("weather_data/alicesprings.xlsx")

darwin = locations.get_group("Darwin")
darwin.to_excel("weather_data/darwin.xlsx")

katherine = locations.get_group("Katherine")
katherine.to_excel("weather_data/katherine.xlsx")

uluru = locations.get_group("Uluru")
uluru.to_excel("weather_data/uluru.xlsx")