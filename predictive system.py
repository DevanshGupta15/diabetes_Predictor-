# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 

loaded_model =pickle.load(open('/Users/devansh/Desktop/Delploying ml /trained_model2.sav','rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_to_array = np.asarray(input_data)
input_data_reshaped = input_data_to_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
