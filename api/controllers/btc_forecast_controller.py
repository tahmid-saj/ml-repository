from api.services.mongodb import create_mongodb_connection, close_mongodb_connection
from api.models.btc_forecast.btc_forecast_models import DailyPrediction, TwoWeekPrediction
from utils.constants.btc_forecast_constants \
  import BTC_FORECAST_DB_NAME, BTC_FORECAST_DAILY_PREDICTION_COLLECTION, BTC_FORECAST_2_WEEK_PREDICTION_COLLECTION

from datetime import datetime, timedelta

def saveDailyPrediction(prediction_result):
  # create connection
  mongodb_client_connection = create_mongodb_connection()
  btc_forecast_db = mongodb_client_connection[BTC_FORECAST_DB_NAME]
  collection = btc_forecast_db[BTC_FORECAST_DAILY_PREDICTION_COLLECTION]
  
  # format data
  current_date = datetime.today()
  prediction_date = current_date + timedelta(days=1)
  prediction_price = []
  for prediction in prediction_result: prediction_price.append(float(prediction))
  prediction_date = prediction_date[0]
  
  daily_prediction = DailyPrediction(
    current_date=current_date, 
    prediction_date=prediction_date,
    prediction_price=prediction_price
  )
  formatted_document = daily_prediction.get_formatted_document()
  
  # save data
  inserted_document = collection.insert_one(formatted_document)
  print(f"Inserted mongodb document with ID: {inserted_document.inserted_id}")
  
  # close connection
  close_mongodb_connection(mongodb_client_connection)
  
def save2WeeksPrediction(prediction_result):
  mongodb_client_connection = create_mongodb_connection()
  # create connection
  btc_forecast_db = mongodb_client_connection[BTC_FORECAST_DB_NAME]
  collection = btc_forecast_db[BTC_FORECAST_2_WEEK_PREDICTION_COLLECTION]
  
  # format data
  current_date = datetime.today().strftime("%Y-%m-%d")
  prediction_prices = []
  for prediction in prediction_result: prediction_prices.append(float(prediction))
  
  two_week_prediction = TwoWeekPrediction(
    current_date=current_date, 
    predictions=prediction_prices
  )
  formatted_document = two_week_prediction.get_formatted_document()
  
  # save data
  inserted_document = collection.insert_one(formatted_document)
  print(f"Inserted mongodb document with ID: {inserted_document.inserted_id}")
  
  # close connection
  close_mongodb_connection(mongodb_client_connection)