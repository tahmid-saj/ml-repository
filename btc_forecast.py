from scripts.full_pipeline.btc_forecast_multivariate_current_day import main_btc_forecast_multivariate_current_day
from scripts.full_pipeline.btc_forecast_multivariate_2_weeks import main_forecase_multivariate_2_weeks
from api.controllers.btc_forecast_controller import *

future_forecast_current_day = main_btc_forecast_multivariate_current_day()
future_furecast_2_weeks = main_forecase_multivariate_2_weeks()

saveDailyPrediction(future_forecast_current_day)
save2WeeksPrediction(future_furecast_2_weeks)
