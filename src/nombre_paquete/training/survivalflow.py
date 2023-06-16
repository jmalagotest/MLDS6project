######################################################################################################################################################################################
#                                                                 Tensorflow function for survival analysis                                                                          #
#                                                                     by: Juan Sebastian Malagón Torres                                                                              #
######################################################################################################################################################################################


#packages:
import os
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import keras
import numpy as np


#################################################################################### Utils #######################################################################################

#CompetingSurvivalDataGenerator:
#In progress
##Inputs: 
#   *In progress
##Outputs: 
#   *In progress

class CompetingSurvivalDataGenerator(keras.utils.Sequence):
  def __init__(self, Features, Labels, base_model, batch_size=None, surv_path='global', shuffle=True):
    self.Features = Features
    self.Labels = Labels
    self.base_model = base_model
    self.batch_size = batch_size
    self.surv_path = surv_path
    self.shuffle = shuffle

    self.indexes = np.arange(0,len(self.Features))
    if self.batch_size == None:
      self.batch_size = len(self.Features)

    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.Features) / self.batch_size))
    
  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    X, y = self.__data_generation(indexes)

    return X, y

  def on_epoch_end(self):
    hazards = self.base_model(self.Features)

    buffer = []
    for i in range (hazards.shape[1]):
      event_labels = self.Labels.copy()
      event_labels[:,1] = event_labels[:,1]==(i+1)
      Times ,CumulativeRisk, Survival = Breslow(event_labels, hazards[:,i])
      buffer.append(np.array([Times ,CumulativeRisk, Survival]).T)
    if self.surv_path == 'global':
      global CompetingBasalSurvData, BasalSurvData 
      CompetingBasalSurvData = buffer
      BasalSurvData = buffer[0]
    else:
      with open(self.surv_path, 'wb') as file:
        pickle.dump(buffer, file)
    
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, indexes):
    X = self.Features[indexes]
    y = self.Labels[indexes]
    
    return X, y
    

#SurvivalDataGenerator:
#In progress
##Inputs: 
#   *In progress
##Outputs: 
#   *In progress

class SurvivalDataGenerator(keras.utils.Sequence):
  def __init__(self, Features, Labels, base_model, batch_size=None, surv_path='global', shuffle=True):
    self.Features = Features
    self.Labels = Labels
    self.base_model = base_model
    self.batch_size = batch_size
    self.surv_path = surv_path
    self.shuffle = shuffle

    self.indexes = np.arange(0,len(self.Features))
    if self.batch_size == None:
      self.batch_size = len(self.Features)

    self.on_epoch_end()

  def __len__(self):
    return int(np.floor(len(self.Features) / self.batch_size))
    
  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    X, y = self.__data_generation(indexes)

    return X, y

  def on_epoch_end(self):
    hazards = self.base_model(self.Features)
    Times ,CumulativeRisk, Survival = Breslow(self.Labels, hazards)

    if self.surv_path == 'global':
      global BasalSurvData 
      BasalSurvData = np.array([Times ,CumulativeRisk, Survival]).T
    else:
      with open(self.surv_path, 'wb') as file:
        pickle.dump(np.array([Times ,CumulativeRisk, Survival]).T, file)
    
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, indexes):
    X = self.Features[indexes]
    y = self.Labels[indexes]
    
    return X, y

#################################################################################### Cox model #######################################################################################

#LogNegativePartialLikelihood:
#A implementation of common used loss function for cox model
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *loss(float): estimated log negative partial likelihood

def LogNegativePartialLikelihood(labels, hazards):
  labels = tf.cast(labels, dtype=tf.float64)
  hazards = tf.cast(hazards, dtype=tf.float64)
  times = labels [:,0]
  events = labels [:,1]

  n = tf.shape(hazards)[0]
  
  loss = tf.constant([0], dtype=tf.float64)
  n_events = tf.constant([0], dtype=tf.float64)
  for i in range (n):
    patient_time = times[i]
    patient_event = events[i]
    patient_hazard = hazards[i]

    global_risk = tf.math.log((tf.math.reduce_sum(tf.math.exp(hazards[times >= patient_time]))))
    patient_loss = patient_event * (patient_hazard - global_risk)
    n_events = n_events + patient_event
    loss = loss + patient_loss

  loss = (-1/n_events)*loss
  return(loss)
  
  
#Breslow:
#A implementation of Breslow method for basal survival estimation
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *times(array): An array with the different times reported in the experiment (numer of falls).
#   *cumulative_risk (array): An array with the estimated cumulative risk in each time point.
#   *survival (array): An array with the estimated survival probability in each time point.

def Breslow(labels, hazards):
  labels = tf.cast(labels, dtype=tf.float32)
  hazards = tf.cast(hazards, dtype=tf.float32)

  times = labels[:,0]
  events = labels[:,1]

  n = tf.shape(times)[0]

  index = tf.argsort(times)
  times = tf.gather(times, index)
  events = tf.gather(events, index)
  hazards = tf.gather(hazards, index)

  repeated_times = tf.repeat(tf.reshape(times,[n,1]), n, axis=1)
  hazard_selection_mask = tf.cast((tf.transpose(repeated_times) - repeated_times)>=0, tf.float32)

  repeated_hazards = tf.repeat(tf.reshape(hazards,[1,n]), n, axis=0)
  risk = events / tf.math.reduce_sum(tf.math.exp(repeated_hazards) * hazard_selection_mask,1)

  repeated_risk = tf.repeat(tf.reshape(risk,[n,1]), n, axis=1)
  triangular_mask = tfp.math.fill_triangular(tf.ones(tf.cast(((n*n)-n)/2,tf.int32)+n), upper=True)
  cumulative_risk = tf.math.reduce_sum(repeated_risk * triangular_mask,0)
  survival = tf.math.exp(-1*cumulative_risk)

  return(times, cumulative_risk, survival)
  
  
#harrell_index:
#A implementation of Harrel's concordance index 
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *C_index(float): The C index between the times and the estimated hazards.

def harrell_index(labels, hazards):
  times = labels [:,0]
  event = labels [:,1]
  
  n = tf.shape(times)[0]
  
  hazards = tf.reshape(hazards, shape=(n,))
  risk = tf.math.exp(hazards)

  NUM = 0
  DEN = 0
  for i in range (n):
    current_time = times[i]
    current_risk = risk[i]
    
    num = tf.cast(current_risk < risk, tf.int32) * tf.cast(current_time > times, tf.int32) * tf.cast(event, tf.int32)
    num = tf.math.reduce_sum(num)

    den = tf.cast(current_time > times, tf.int32) * tf.cast(event, tf.int32)
    den = tf.math.reduce_sum(den)

    NUM = NUM + num
    DEN = DEN + den

  C_index = NUM/DEN
  return(C_index)


#Time_AUC:
#An implementation of time dependent ROC
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *mean_auc(float): The mean of the different AUC computed for each time

def time_AUC(labels, hazards, surv_path = 'global'):
  hazards = tf.transpose(hazards)[0]
  auc = tf.keras.metrics.AUC(num_thresholds=100)
  
  labels = tf.cast(labels, dtype=tf.float32)
  hazards = tf.cast(hazards, dtype=tf.float32)
  
  Npatients = tf.shape(hazards)[0]

  if surv_path == 'global':
    global BasalSurvData 
    surv_data = BasalSurvData.copy()
  else:
    with open(surv_path, 'rb') as file:
      surv_data = pickle.load(file)

  times_ ,ch, S = surv_data[:,0], surv_data[:,1], surv_data[:,2]

  times = labels[:,0]
  events = labels[:,1]

  unique_times = tf.unique(times[events==1])[0]

  mean_AUC = tf.constant(0,tf.float32)
  for time in unique_times:
    approach_time = times_[tf.math.argmin(tf.math.abs(times_ - time))]
    BasalCH_t = tf.math.reduce_max(ch[times_ == approach_time])
    CH_t = (tf.ones(Npatients)* BasalCH_t) * tf.math.exp(hazards)
    SurvProbs = tf.math.exp(-1*CH_t)
    state = tf.cast(times <= time,tf.float32)
    slection_mask = state*(1-events)

    SurvProbs = 1-SurvProbs[slection_mask==0]
    state = state[slection_mask==0]
    auc.reset_state()
    auc.update_state(state, SurvProbs)
    mean_AUC = mean_AUC + auc.result()

  return(mean_AUC/tf.cast(tf.shape(unique_times)[0],tf.float32))
  
  
################################################################################# Competing model ########################################################################################

#CompetingLogNegativePartialLikelihood:
#A implementation of common used loss function for cox model
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *loss(float): estimated log negative partial likelihood

def CompetingLogNegativePartialLikelihood(labels, hazards, weights):
  labels = tf.cast(labels, dtype=tf.float64)
  hazards = tf.cast(hazards, dtype=tf.float64)
  times = labels [:,0]
  events = labels [:,1]
  unique_values, idx = tf.unique(events)
  unique_values = tf.sort(unique_values, direction='ASCENDING')
  if 0 == tf.math.reduce_min(unique_values):
    unique_values = unique_values[1:]

  event_types = tf.shape(unique_values)[0]

  mean_loss = tf.constant([0],tf.float64)
  for i in range (event_types):
    event_id = unique_values[i]

    event_= tf.cast(events == event_id, tf.float64)
    labels_ = tf.transpose(tf.concat([[times],[event_]],0))
    w = weights[i]
    
    event_loss = LogNegativePartialLikelihood(labels_, hazards[:,i])
    mean_loss = mean_loss + (event_loss * w)

  return(mean_loss)
  

#harrell_index:
#A implementation of Harrel's concordance index 
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *C_index(float): The C index between the times and the estimated hazards.

def mean_harrell_index(labels, hazards):
  labels = tf.cast(labels, dtype=tf.float64)
  hazards = tf.cast(hazards, dtype=tf.float64)
  times = labels [:,0]
  events = labels [:,1]
  unique_values, idx = tf.unique(events)
  unique_values = tf.sort(unique_values, direction='ASCENDING')
  if 0 == tf.math.reduce_min(unique_values):
    unique_values = unique_values[1:]

  event_types = tf.shape(unique_values)[0]

  mean_c_index = tf.constant([0], tf.float64)
  for i in range (event_types):
    event_id = unique_values[i]

    event_= tf.cast(events == event_id, tf.float64)
    labels_ = tf.transpose(tf.concat([[times],[event_]],0))
    
    c_index = harrell_index(labels_, hazards[:,i])
    mean_c_index = mean_c_index + c_index

  mean_c_index = mean_c_index/tf.cast(event_types, tf.float64)
  return(mean_c_index)
  
  
#Time_AUC:
#An implementation of time dependent ROC
##Inputs: 
#   *labels(array): An nx2 array with the labels, the columns correspond to the time and event indicator IN THIS ORDER.
#   *hazrads(array): A vector whit the linear combination of the cox parameters and the covariables.
##Outputs: 
#   *mean_auc(float): The mean of the different AUC computed for each time

def macro_time_AUC(labels, hazards, SurvDataPaths='global'):
  labels = tf.cast(labels, dtype=tf.float64)
  hazards = tf.cast(hazards, dtype=tf.float64)
  times = labels [:,0]
  events = labels [:,1]
  unique_values, idx = tf.unique(events)
  unique_values = tf.sort(unique_values, direction='ASCENDING')

  if 0 == tf.math.reduce_min(unique_values):
    unique_values = unique_values[1:]

  event_types = tf.shape(unique_values)[0]
  
  if SurvDataPaths == 'global':
      global CompetingBasalSurvData
      SurvDatas = CompetingBasalSurvData.copy()

  mean_AUC = tf.constant([0], tf.float32)
  for i in range (event_types):
    event_id = unique_values[i]

    event_= tf.cast(events == event_id, tf.float64)
    labels_ = tf.transpose(tf.concat([[times],[event_]],0))

    if SurvDataPaths == 'global':
        global BasalSurvData
        BasalSurvData = SurvDatas[i]
        AUC = time_AUC(labels_, tf.transpose([hazards[:,i]]), 'global')
    else:
        AUC = time_AUC(labels_, hazards[:,i], SurvDataPaths[i])
        
    mean_AUC = mean_AUC + AUC

  mean_AUC = mean_AUC/tf.cast(event_types, tf.float32)
  return(mean_AUC)
