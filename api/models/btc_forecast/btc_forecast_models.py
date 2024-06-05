from datetime import datetime, timedelta

class DailyPrediction:
  def __init__(self, current_date, prediction_date, prediction_price):
    self.current_date = current_date
    self.prediction_date = prediction_date
    self.prediction_price = prediction_price
  
  def get_formatted_document(self):
    res_formatted_document = {
      "current_date": self.current_date,
      "prediction": {
        "prediction_date": self.prediction_date,
        "prediction_price": self.prediction_price
      }
    }
    
    return res_formatted_document

class TwoWeekPrediction:
  def __init__(self, current_date, predictions):
    self.current_date = current_date
    self.predictions = predictions
  
  def get_formatted_document(self):
    predictions = []
    beginning_date = datetime.strptime(self.current_date, "%Y-%m-%d")
    for i in range(1, 11): 
      prediction_date = beginning_date + timedelta(days=i)
      predictions.append({
        "prediction_date": prediction_date,
        "prediction_price": self.predictions[i]
      })
    
    res_formatted_document = {
      "current_date": self.current_date,
      "predictions": predictions
    }
    
    return res_formatted_document