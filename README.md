# Using Blood Glucose Data and Deep Learning to Identify Time of Day
Implemented a `TensorFlow` time classifier model which uses blood glucose data
from a Dexcom continuous glucose monitor (CGM) to predict whether or not a
continuous sequence of CGM blood glucose readings are within a certain time
range of the day (ie. between 2am and 8am). The model is a `TensorFlow`
Sequential Conv1D binary classifier model which takes in time series inputs. 
The model predicts whether or not a series of continuous glucose
readings are inside or outside a time range. The user determines the start and
end time of this window. Each series contains a sequence of consecutive data
derived from a CGM reading.

README.md work in progress.
