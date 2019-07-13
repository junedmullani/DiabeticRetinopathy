from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import sys  
mod=load_model('C:/Users/DELL/Desktop/Test WEB DR/model.hd5')
    
test_gen = ImageDataGenerator(rescale = 1./255)


PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
CAPTHA_ROOT = os.path.join(PROJECT_PATH,'test_images')
    
test_data = test_gen.flow_from_directory(CAPTHA_ROOT,
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary', shuffle=True)
    
predicted = mod.predict_generator(test_data)
    
y_pred = predicted[0][0] > 0.4
percent_chance = round(predicted[0][0]*100, 2)
    
print('DR status ==>'+str(y_pred))
print('Predict percentage ==> '+str(percent_chance))



