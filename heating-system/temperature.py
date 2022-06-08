HEAT_TRANSFER_COEFFICIENT = 0.1
TARGET_TEMPERATURE = 23.5
HEAT_REGULATION_COEFFICIENT=0.1
HEAT_POWER_COEFFICIENT=1.4

def get_next_temperature(action, indoor_temperature, outdoor_temperature, granularity):
    next_indoor_temperature = (
            HEAT_TRANSFER_COEFFICIENT *(outdoor_temperature - indoor_temperature)
            + action*HEAT_REGULATION_COEFFICIENT*(TARGET_TEMPERATURE-outdoor_temperature)
            + action*HEAT_POWER_COEFFICIENT*(TARGET_TEMPERATURE-indoor_temperature)
    )*granularity + indoor_temperature
    return next_indoor_temperature



