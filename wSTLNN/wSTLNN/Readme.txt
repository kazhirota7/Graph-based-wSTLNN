The sample code is used to perform classification on the occupancy detection dataset from the UCI Machine Learning Repository. 
The dataset is a time-series dataset that records the information including temperature (s1), relative humidity (s2), light 
intensity (s3), CO2 concentration (s4), humidity radio (s5), and occupancy of an office room from 2015-02-02 to 2015-02-18. 
The data is recorded every minute and the target variable is the occupancy (-1 for not occupied and 1 for occupied).

You can run the "wSTLNN.py" to see the classification result, where the wSTL-NN model is designed in "Model.py". The formula structure in 
the "Model.py" is a weighted STL formula with the structure of G_{[k_1,k_2]}^w tx>=b. If you need to design other formula structures, please 
modify the class TL_NN according to the structure you need and the activation function provided in the paper "Neural Network for Weighted 
Signal Temporal Logic".