import datetime
import numpy as N

class FruitingBranch: ...

class Simulation:
    climate: list
    cultivar_parameters: list
    start_date: datetime.date
    version: int
    year: int
    site_parameters: list
    def __init__(self, version: int = 0x0400, **kwargs): ...

class Soil: ...

class SoilInit:
    lyrsol: int
    def __init__(self, initial, hydrology): ...

class State:
    actual_transpiration: float
    average_temperature: float
    date: datetime.date
    day_length: float
    daytime_temperature: float
    deep_soil_temperature: float
    evapotranspiration: float
    kday: int
    hours: N.ndarray
    leaf_area_index: float
    light_interception: float
    max_leaf_water_potential: float
    min_leaf_water_potential: float
    nighttime_temperature: float
    number_of_green_bolls: float
    number_of_open_bolls: float
    number_of_vegetative_branches: int
    number_of_pre_fruiting_nodes: int
    plant_height: float
    pre_fruiting_leaf_area: list[float]
    pre_fruiting_nodes_age: list[float]
    soil_temperature: N.ndarray
    vegetative_branches: list[VegetativeBranch]
    water_stress: float
    def apply_fertilizer(self, row_space: float, plant_population: float): ...
    def average_psi(self, row_space: float): ...
    def leaf_water_potential(self, row_space: float): ...
    def water_uptake(self, row_space: float, per_plant_area: float): ...

class VegetativeBranch:
    fruiting_branches: list
