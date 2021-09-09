The sample code "CaseStudy1" is used to perform classification on the rainfall dataset from the Australian Government Bureau of Meteorology. The dataset is a
time-series dataset that records information including  minimum  temperature(g1),  maximum  temperature  (g2),  amount  of  rainfall  (g3),evaporation (g4), sunshine (g5),
wind gust speed (g6), windspeed  at  9am  (g7),  wind  speed  at  3pm  (g8),  humidity  at9am  (g9),  humidity  at  3pm  (g10),  pressure  at  9am  (g11),
pressure  at  3pm  (g12),  cloud  at  9am  (g13),  cloud  at  3pm(g14),  temperature  at  9am  (g15),  temperature  at  3pm  (g16),and  whether  there  was
any  rain  on  the  given  day  (-1  or  1)(g17) for 49 regions in Australia. We construct graph structures by considering any two regions within a 300km
radius to be neighbors. The target variable is -1 for no rain on the following day and 1 for rain on following day.

The sample code "CaseStudy2" is used to perform classification on a simulated COVID-19 dataset from the DiBernardo Group Repository.
The dataset is a time-series dataset that records information including the percentage of people infected (g1), quarantined (g2),
deceased (g3), and hospitalized  due  to  COVID-19 (g4). We construct a graph structure using the edges provided by the DiBernardo
Group Repository. The target variable classifies into -1 for "strict lockdown measures" and 1 for "No lockdown measures". The performance is evaluated in
comparison to K nearest neighbors and decision tree, which you can see the result by running "other_classifiers.py".

You can run the "CaseStudy1.py" and "CaseStudy2.py" to see the classification result for rainfall prediction and COVID lockdown prediction, respectively.
The w-GSTL models are designed in "weatherModel.py" and "COVIDModel.py", and the operators are learned using "weatherOpModel.py" and "COVIDOpModel.py".
To learn other formula structures, please modify the forward method in the neural network models.