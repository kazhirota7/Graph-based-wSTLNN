# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('AustWeather_zeros.csv', dtype={'Date':str, 'Location':str, 'MinTemp':float, 'MaxTemp':float, 'Rainfall':float, 'Evaporation':float, 'Sunshine':float, 'WindGustSpeed':float, 'WindSpeed9am':float, 'WindSpeed3pm':float, 'Pressure9am':float, 'Pressure3pm':float, 'Cloud9am':float, 'Cloud3pm':float, 'Temp9am':float, 'Temp3pm':float, 'RainToday':str, "RainTomorrow":str})

locations = df.groupby(df.Location)
albury = locations.get_group("Albury")

albury_train = albury.iloc[1462:2724,]
albury_test = albury.iloc[2724:,]
albury_train.to_excel("weather_data/albury_train.xlsx")
albury_test.to_excel("weather_data/albury_test.xlsx")
print("albury",str(len(albury_train)))
print("albury",str(len(albury_test)))

badgeryscreek = locations.get_group("BadgerysCreek")
badgeryscreek_train = badgeryscreek.iloc[1431:2693,]
badgeryscreek_test = badgeryscreek.iloc[2693:,]
badgeryscreek_train.to_excel("weather_data/badgeryscreek_train.xlsx")
badgeryscreek_test.to_excel("weather_data/badgeryscreek_test.xlsx")
print("badgery",len(badgeryscreek_train))
print("badgery",len(badgeryscreek_test))

cobar = locations.get_group("Cobar")
cobar_train = cobar.iloc[1431:2693,]
cobar_test = cobar.iloc[2693:,]
cobar_train.to_excel("weather_data/cobar_train.xlsx")
cobar_test.to_excel("weather_data/cobar_test.xlsx")
print("cobar",str(len(cobar_train)))
print("cobar",str(len(cobar_test)))

coffsharbour = locations.get_group("CoffsHarbour")
coffsharbour_train = coffsharbour.iloc[1431:2693,]
coffsharbour_test = coffsharbour.iloc[2693:,]
coffsharbour_train.to_excel("weather_data/coffsharbour_train.xlsx")
coffsharbour_test.to_excel("weather_data/coffsharbour_test.xlsx")
print("coffsharbour",str(len(coffsharbour_train)))
print("coffsharbour",str(len(coffsharbour_test)))


moree = locations.get_group("Moree")
moree_train = moree.iloc[1431:2693,]
moree_test = moree.iloc[2693:,]
moree_train.to_excel("weather_data/moree_train.xlsx")
moree_test.to_excel("weather_data/moree_test.xlsx")
print("moree",str(len(moree_train)))
print("moree",str(len(moree_test)))

newcastle = locations.get_group("Newcastle")
newcastle_train = newcastle.iloc[1462:2724,]
newcastle_test = newcastle.iloc[2724:,]
newcastle_train.to_excel("weather_data/newcastle_train.xlsx")
newcastle_test.to_excel("weather_data/newcastle_test.xlsx")
print("newcastle",str(len(newcastle_train)))
print("newcastle",str(len(newcastle_test)))

norahhead = locations.get_group("0rahHead")
norahhead_train = norahhead.iloc[1431:2693,]
norahhead_test = norahhead.iloc[2693:,]
norahhead_train.to_excel("weather_data/norahhead_train.xlsx")
norahhead_test.to_excel("weather_data/norahhead_test.xlsx")
print("norahhead",str(len(norahhead_train)))
print("norahhead",str(len(norahhead_test)))

norfolkisland = locations.get_group("0rfolkIsland")
norfolkisland_train = norfolkisland.iloc[1431:2693,]
norfolkisland_test = norfolkisland.iloc[2693:,]
norfolkisland_train.to_excel("weather_data/norfolkisland_train.xlsx")
norfolkisland_test.to_excel("weather_data/norfolkisland_test.xlsx")
print("norfolkisland",str(len(norfolkisland_train)))
print("norfolkisland",str(len(norfolkisland_test)))


penrith = locations.get_group("Penrith")
penrith_train = penrith.iloc[1461:2723,]
penrith_test = penrith.iloc[2723:,]
penrith_train.to_excel("weather_data/penrith_train.xlsx")
penrith_test.to_excel("weather_data/penrith_test.xlsx")
print("penrith",str(len(penrith_train)))
print("penrith",str(len(penrith_test)))

richmond = locations.get_group("Richmond")
richmond_train = richmond.iloc[1431:2693,]
richmond_test = richmond.iloc[2693:,]
richmond_train.to_excel("weather_data/richmond_train.xlsx")
richmond_test.to_excel("weather_data/richmond_test.xlsx")
print("richmond",str(len(richmond_train)))
print("richmond",str(len(richmond_test)))

sydney = locations.get_group("Sydney")
sydney_train = sydney.iloc[1766:3028,]
sydney_test = sydney.iloc[3028:,]
sydney_train.to_excel("weather_data/sydney_train.xlsx")
sydney_test.to_excel("weather_data/sydney_test.xlsx")
print("sydney",len(sydney_train))
print("sydney",len(sydney_test))

sydneyairport = locations.get_group("SydneyAirport")
sydneyairport_train = sydneyairport.iloc[1431:2693,]
sydneyairport_test = sydneyairport.iloc[2693:,]
sydneyairport_train.to_excel("weather_data/sydneyairport_train.xlsx")
sydneyairport_test.to_excel("weather_data/sydneyairport_test.xlsx")
print("sydneyairport",str(len(sydneyairport_train)))
print("sydneyairport",str(len(sydneyairport_test)))

waggawagga = locations.get_group("WaggaWagga")
waggawagga_train = waggawagga.iloc[1431:2693,]
waggawagga_test = waggawagga.iloc[2693:,]
waggawagga_train.to_excel("weather_data/waggawagga_train.xlsx")
waggawagga_test.to_excel("weather_data/waggawagga_test.xlsx")
print("waggawagga",str(len(waggawagga_train)))
print("waggawagga",str(len(waggawagga_test)))


williamtown = locations.get_group("Williamtown")
williamtown_train = williamtown.iloc[1431:2693,]
williamtown_test = williamtown.iloc[2693:,]
williamtown_train.to_excel("weather_data/williamtown_train.xlsx")
williamtown_test.to_excel("weather_data/williamtown_test.xlsx")
print("williamtown",str(len(williamtown_train)))
print("williamtown",str(len(williamtown_test)))

wollongong = locations.get_group("Wollongong")
wollongong_train = wollongong.iloc[1462:2724,]
wollongong_test = wollongong.iloc[2724:,]
wollongong_train.to_excel("weather_data/wollongong_train.xlsx")
wollongong_test.to_excel("weather_data/wollongong_test.xlsx")
print("wollongong",str(len(wollongong_train)))
print("wollongong",str(len(wollongong_test)))

canberra = locations.get_group("Canberra")
canberra_train = canberra.iloc[1858:3120,]
canberra_test = canberra.iloc[3120:,]
canberra_train.to_excel("weather_data/canberra_train.xlsx")
canberra_test.to_excel("weather_data/canberra_test.xlsx")
print("canberra", str(len(canberra_train)))
print("canberra", str(len(canberra_test)))

tuggeranong = locations.get_group("Tuggera0ng")
tuggeranong_train = tuggeranong.iloc[1461:2723,]
tuggeranong_test = tuggeranong.iloc[2723:,]
tuggeranong_train.to_excel("weather_data/tuggeranong_train.xlsx")
tuggeranong_test.to_excel("weather_data/tuggeranong_test.xlsx")
print("tuggeranong",str(len(tuggeranong_train)))
print("tuggeranong",str(len(tuggeranong_test)))

mountginini = locations.get_group("MountGinini")
mountginini_train = mountginini.iloc[1462:2724,]
mountginini_test = mountginini.iloc[2724:,]
mountginini_train.to_excel("weather_data/mountginini_train.xlsx")
mountginini_test.to_excel("weather_data/mountginini_test.xlsx")
print("mountginini",str(len(mountginini_train)))
print("mountginini",str(len(mountginini_test)))

ballarat = locations.get_group("Ballarat")
ballarat_train = ballarat.iloc[1462:2724,]
ballarat_test = ballarat.iloc[2724:,]
ballarat_train.to_excel("weather_data/ballarat_train.xlsx")
ballarat_test.to_excel("weather_data/ballarat_test.xlsx")
print("ballarat",str(len(ballarat_train)))
print("ballarat",str(len(ballarat_test)))

bendigo = locations.get_group("Bendigo")
bendigo_train = bendigo.iloc[1462:2724,]
bendigo_test = bendigo.iloc[2724:,]
bendigo_train.to_excel("weather_data/bendigo_train.xlsx")
bendigo_test.to_excel("weather_data/bendigo_test.xlsx")
print("bendigo",str(len(bendigo_train)))
print("bendigo",str(len(bendigo_test)))


sale = locations.get_group("Sale")
sale_train = sale.iloc[1431:2693,]
sale_test = sale.iloc[2693:,]
sale_train.to_excel("weather_data/sale_train.xlsx")
sale_test.to_excel("weather_data/sale_test.xlsx")
print("sale",str(len(sale_train)))
print("sale",str(len(sale_test)))

melbourne = locations.get_group("Melbourne")
melbourne_train = melbourne.iloc[1615:2877,]
melbourne_test = melbourne.iloc[2877:,]
melbourne_train.to_excel("weather_data/melbourne_train.xlsx")
melbourne_test.to_excel("weather_data/melbourne_test.xlsx")
print("melbourne", len(melbourne_train))
print("melbourne", len(melbourne_test))

melbourneairport = locations.get_group("MelbourneAirport")
melbourneairport_train = melbourneairport.iloc[1431:2693,]
melbourneairport_test = melbourneairport.iloc[2693:,]
melbourneairport_train.to_excel("weather_data/melbourneairport_train.xlsx")
melbourneairport_test.to_excel("weather_data/melbourneairport_test.xlsx")
print("melbourneairport",str(len(melbourneairport_train)))
print("melbourneairport",str(len(melbourneairport_test)))

mildura = locations.get_group("Mildura")
mildura_train = mildura.iloc[1431:2693,]
mildura_test = mildura.iloc[2693:,]
mildura_train.to_excel("weather_data/mildura_train.xlsx")
mildura_test.to_excel("weather_data/mildura_test.xlsx")
print("mildura",str(len(mildura_train)))
print("mildura",str(len(mildura_test)))

nhil = locations.get_group("Nhil")
nhil_train = nhil.iloc[:1262,]
nhil_test = nhil.iloc[1262:,]
nhil_train.to_excel("weather_data/nhil_train.xlsx")
nhil_test.to_excel("weather_data/nhil_test.xlsx")
print("nhil",len(nhil_train))
print("nhil",len(nhil_test))

portland = locations.get_group("Portland")
portland_train = portland.iloc[1431:2693,]
portland_test = portland.iloc[2693:,]
portland_train.to_excel("weather_data/portland_train.xlsx")
portland_test.to_excel("weather_data/portland_test.xlsx")
print("portland",str(len(portland_train)))
print("portland",str(len(portland_test)))

watsonia = locations.get_group("Watsonia")
watsonia_train = watsonia.iloc[1431:2693,]
watsonia_test = watsonia.iloc[2693:,]
watsonia_train.to_excel("weather_data/watsonia_train.xlsx")
watsonia_test.to_excel("weather_data/watsonia_test.xlsx")
print("watsonia",str(len(watsonia_train)))
print("watsonia",str(len(watsonia_test)))

dartmoor = locations.get_group("Dartmoor")
dartmoor_train = dartmoor.iloc[1431:2693,]
dartmoor_test = dartmoor.iloc[2693:,]
dartmoor_train.to_excel("weather_data/dartmoor_train.xlsx")
dartmoor_test.to_excel("weather_data/dartmoor_test.xlsx")
print("dartmoor",str(len(dartmoor_train)))
print("dartmoor",str(len(dartmoor_test)))

brisbane = locations.get_group("Brisbane")
brisbane_train = brisbane.iloc[1615:2877,]
brisbane_test = brisbane.iloc[2877:,]
brisbane_train.to_excel("weather_data/brisbane_train.xlsx")
brisbane_test.to_excel("weather_data/brisbane_test.xlsx")
print("brisbane",len(brisbane_train))
print("brisbane",len(brisbane_test))

cairns = locations.get_group("Cairns")
cairns_train = cairns.iloc[1462:2724,]
cairns_test = cairns.iloc[2724:,]
cairns_train.to_excel("weather_data/cairns_train.xlsx")
cairns_test.to_excel("weather_data/cairns_test.xlsx")
print("cairns",str(len(cairns_train)))
print("cairns",str(len(cairns_test)))

goldcoast = locations.get_group("GoldCoast")
goldcoast_train = goldcoast.iloc[1462:2724,]
goldcoast_test = goldcoast.iloc[2724:,]
goldcoast_train.to_excel("weather_data/goldcoast_train.xlsx")
goldcoast_test.to_excel("weather_data/goldcoast_test.xlsx")
print("goldcoast",str(len(goldcoast_train)))
print("goldcoast",str(len(goldcoast_test)))

townsville = locations.get_group("Townsville")
townsville_train = townsville.iloc[1462:2724,]
townsville_test = townsville.iloc[2724:,]
townsville_train.to_excel("weather_data/townsville_train.xlsx")
townsville_test.to_excel("weather_data/townsville_test.xlsx")
print("townsville",str(len(townsville_train)))
print("townsville",str(len(townsville_test)))

adelaide = locations.get_group("Adelaide")
adelaide_train = adelaide.iloc[1615:2877,]
adelaide_test = adelaide.iloc[2877:,]
adelaide_train.to_excel("weather_data/adelaide_train.xlsx")
adelaide_test.to_excel("weather_data/adelaide_test.xlsx")
print("adelaide",str(len(adelaide_train)))
print("adelaide",str(len(adelaide_test)))

mountgambier = locations.get_group("MountGambier")
mountgambier_train = mountgambier.iloc[1462:2724,]
mountgambier_test = mountgambier.iloc[2724:,]
mountgambier_train.to_excel("weather_data/mountgambier_train.xlsx")
mountgambier_test.to_excel("weather_data/mountgambier_test.xlsx")
print("mountgambier",str(len(mountgambier_train)))
print("mountgambier",str(len(mountgambier_test)))

nuriootpa = locations.get_group("Nuriootpa")
nuriootpa_train = nuriootpa.iloc[1431:2693,]
nuriootpa_test = nuriootpa.iloc[2693:,]
nuriootpa_train.to_excel("weather_data/nuriootpa_train.xlsx")
nuriootpa_test.to_excel("weather_data/nuriootpa_test.xlsx")
print("nuriootpa",str(len(nuriootpa_train)))
print("nuriootpa",str(len(nuriootpa_test)))

woomera = locations.get_group("Woomera")
woomera_train = woomera.iloc[1431:2693,]
woomera_test = woomera.iloc[2693:,]
woomera_train.to_excel("weather_data/woomera_train.xlsx")
woomera_test.to_excel("weather_data/woomera_test.xlsx")
print("woomera",str(len(woomera_train)))
print("woomera",str(len(woomera_test)))

albany = locations.get_group("Albany")
albany_train = albany.iloc[1462:2724,]
albany_test = albany.iloc[2724:,]
albany_train.to_excel("weather_data/albany_train.xlsx")
albany_test.to_excel("weather_data/albany_test.xlsx")
print("albany",str(len(albany_train)))
print("albany",str(len(albany_test)))

witchcliffe = locations.get_group("Witchcliffe")
witchcliffe_train = witchcliffe.iloc[1431:2693,]
witchcliffe_test = witchcliffe.iloc[2693:,]
witchcliffe_train.to_excel("weather_data/witchcliffe_train.xlsx")
witchcliffe_test.to_excel("weather_data/witchcliffe_test.xlsx")
print("witchcliffe",str(len(witchcliffe_train)))
print("witchcliffe",str(len(witchcliffe_test)))

pearceraaf = locations.get_group("PearceRAAF")
pearceraaf_train = pearceraaf.iloc[1431:2693,]
pearceraaf_test = pearceraaf.iloc[2693:,]
pearceraaf_train.to_excel("weather_data/pearceraaf_train.xlsx")
pearceraaf_test.to_excel("weather_data/pearceraaf_test.xlsx")
print("pearceraaf",str(len(pearceraaf_train)))
print("pearceraaf",str(len(pearceraaf_test)))

perthairport = locations.get_group("PerthAirport")
perthairport_train = perthairport.iloc[1431:2693,]
perthairport_test = perthairport.iloc[2693:,]
perthairport_train.to_excel("weather_data/perthairport_train.xlsx")
perthairport_test.to_excel("weather_data/perthairport_test.xlsx")
print("perthairport",str(len(perthairport_train)))
print("perthairport",str(len(perthairport_test)))

perth = locations.get_group("Perth")
perth_train = perth.iloc[1615:2877,]
perth_test = perth.iloc[2877:,]
perth_train.to_excel("weather_data/perth_train.xlsx")
perth_test.to_excel("weather_data/perth_test.xlsx")
print("perth",str(len(perth_train)))
print("perth",str(len(perth_test)))

salmongums = locations.get_group("SalmonGums")
salmongums_train = salmongums.iloc[1423:2685,]
salmongums_test = salmongums.iloc[2685:,]
salmongums_train.to_excel("weather_data/salmongums_train.xlsx")
salmongums_test.to_excel("weather_data/salmongums_test.xlsx")
print("salmongums",str(len(salmongums_train)))
print("salmongums",str(len(salmongums_test)))

walpole = locations.get_group("Walpole")
walpole_train = walpole.iloc[1428:2690,]
walpole_test = walpole.iloc[2690:,]
walpole_train.to_excel("weather_data/walpole_train.xlsx")
walpole_test.to_excel("weather_data/walpole_test.xlsx")
print("walpole",str(len(walpole_train)))
print("walpole",str(len(walpole_test)))

hobart = locations.get_group("Hobart")
hobart_train = hobart.iloc[1615:2877,]
hobart_test = hobart.iloc[2877:,]
hobart_train.to_excel("weather_data/hobart_train.xlsx")
hobart_test.to_excel("weather_data/hobart_test.xlsx")
print("hobart",str(len(hobart_train)))
print("hobart",str(len(hobart_test)))

launceston = locations.get_group("Launceston")
launceston_train = launceston.iloc[1462:2724,]
launceston_test = launceston.iloc[2724:,]
launceston_train.to_excel("weather_data/launceston_train.xlsx")
launceston_test.to_excel("weather_data/launceston_test.xlsx")
print("launceston",str(len(launceston_train)))
print("launceston",str(len(launceston_test)))

alicesprings = locations.get_group("AliceSprings")
alicesprings_train = alicesprings.iloc[1462:2724,]
alicesprings_test = alicesprings.iloc[2724:,]
alicesprings_train.to_excel("weather_data/alicesprings_train.xlsx")
alicesprings_test.to_excel("weather_data/alicesprings_test.xlsx")
print("alicesprings",str(len(alicesprings_train)))
print("alicesprings",str(len(alicesprings_test)))

darwin = locations.get_group("Darwin")
darwin_train = darwin.iloc[1615:2877,]
darwin_test = darwin.iloc[2877:,]
darwin_train.to_excel("weather_data/darwin_train.xlsx")
darwin_test.to_excel("weather_data/darwin_test.xlsx")
print("darwin",str(len(darwin_train)))
print("darwin",str(len(darwin_test)))

katherine = locations.get_group("Katherine")
katherine_train = katherine.iloc[:1262,]
katherine_test = katherine.iloc[1262:,]
katherine_train.to_excel("weather_data/katherine_train.xlsx")
katherine_test.to_excel("weather_data/katherine_test.xlsx")
print("katherine",str(len(katherine_train)))
print("katherine",str(len(katherine_test)))

uluru = locations.get_group("Uluru")
uluru_train = uluru.iloc[:1262,]
uluru_test = uluru.iloc[1262:,]
uluru_train.to_excel("weather_data/uluru_train.xlsx")
uluru_test.to_excel("weather_data/uluru_test.xlsx")
print("uluru",str(len(uluru_train)))
print("uluru",str(len(uluru_test)))