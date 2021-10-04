import csv
import json
from pathlib import Path

from pytest import fixture

from cotton2k.core import run
from cotton2k.core.simulation import Simulation


@fixture
def empty_json(tmp_path: Path) -> Path:
    empty = tmp_path / "empty.cotton2k.json"
    empty.write_text(json.dumps({}))
    return empty


@fixture
def test_json(tmp_path: Path) -> Path:
    test = tmp_path / "alaer.cotton2k.json"
    profile = {
        "description": "HAZOR 1984 experiment, treatment KB1",
        "start_date": "1984-04-01",
        "stop_date": "1984-09-28",
        "emerge_date": "1984-04-08",
        "longitude": 35,
        "latitude": 32,
        "elevation": 50,
        "site_parameters": [
            0,
            1.0,
            4.0,
            2.0,
            0.216,
            20,
            24,
            110,
            1.0,
            24,
            4.0,
            180,
            -2.436,
            0.820,
            -0.930,
            0.30,
            0.18,
        ],
        "row_space": 96.520,
        "skip_row_width": 0.00,
        "plants_per_meter": 10,
        "cultivar_parameters": [
            0,
            0.050,
            0.28,
            0.012,
            0.50,
            1.60,
            0.010,
            20.0,
            0.14,
            27.5,
            0.3293,
            9.482,
            0.30,
            0.040,
            0.01,
            2.8,
            1.44,
            0.40,
            0.110,
            0.40,
            0.12,
            17.0,
            -2.6,
            0.10,
            0.15,
            2.30,
            1.08,
            0.88,
            2.30,
            1.44,
            0.98,
            2.55,
            1.50,
            1.20,
            0.04,
            -31.0,
            -52.50,
            4.50,
            2.50,
            -292.0,
            1.00,
            54.56,
            0.6755,
            0.58,
            0.42,
            0.32,
            0.08,
            5.0,
            1.48,
            8.0,
            0.80,
        ],
        "soil": {
            "initial": [
                {
                    "ammonium_nitrogen": 13.50,
                    "nitrate_nitrogen": 20.00,
                    "organic_matter": 1.200,
                    "water": 90,
                },
                {
                    "ammonium_nitrogen": 13.50,
                    "nitrate_nitrogen": 16.00,
                    "organic_matter": 1.000,
                    "water": 95,
                },
                {
                    "ammonium_nitrogen": 9.000,
                    "nitrate_nitrogen": 12.00,
                    "organic_matter": 0.800,
                    "water": 95,
                },
                {
                    "ammonium_nitrogen": 9.000,
                    "nitrate_nitrogen": 12.00,
                    "organic_matter": 0.600,
                    "water": 95,
                },
                {
                    "ammonium_nitrogen": 6.000,
                    "nitrate_nitrogen": 5.600,
                    "organic_matter": 0.000,
                    "water": 90,
                },
                {
                    "ammonium_nitrogen": 6.000,
                    "nitrate_nitrogen": 5.600,
                    "organic_matter": 0.000,
                    "water": 85,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 5.600,
                    "organic_matter": 0.000,
                    "water": 80,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 5.600,
                    "organic_matter": 0.000,
                    "water": 75,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 75,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 75,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 70,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 70,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 70,
                },
                {
                    "ammonium_nitrogen": 0.000,
                    "nitrate_nitrogen": 0.000,
                    "organic_matter": 0.000,
                    "water": 70,
                },
            ],
            "hydrology": {
                "ratio_implicit": 0.75,
                "max_conductivity": 0.002,
                "field_capacity_water_potential": -0.300,
                "immediate_drainage_water_potential": -0.200,
                "layers": [
                    {
                        "depth": 62.0,
                        "air_dry": 0.2843,
                        "theta": 0.4566,
                        "alpha": 0.0010,
                        "beta": 1.4678,
                        "saturated_hydraulic_conductivity": 0.000,
                        "field_capacity_hydraulic_conductivity": 0.020,
                        "bulk_density": 1.440,
                        "clay": 60.0,
                        "sand": 20.0,
                    },
                    {
                        "depth": 201.0,
                        "air_dry": 0.2613,
                        "theta": 0.4415,
                        "alpha": 0.0018,
                        "beta": 1.4186,
                        "saturated_hydraulic_conductivity": 0.000,
                        "field_capacity_hydraulic_conductivity": 0.020,
                        "bulk_density": 1.480,
                        "clay": 55.0,
                        "sand": 30.0,
                    },
                ],
            },
        },
        "agricultural_inputs": [
            {"type": "fertilization", "date": "1984-04-01", "ammonium": 150.0},
            {"type": "irrigation", "date": "1984-06-22", "amount": 89.000},
            {"type": "irrigation", "date": "1984-07-12", "amount": 104.000},
            {"type": "irrigation", "date": "1984-07-26", "amount": 104.000},
            {"type": "irrigation", "date": "1984-08-08", "amount": 98.000},
            {"type": "irrigation", "date": "1984-08-22", "amount": 72.000},
            {
                "type": "defoliation prediction",
                "date": "1984-09-15",
                "method": 85,
            },
        ],
    }
    with open(Path(__file__).parent / "hazor84.csv") as f:
        profile["climate"] = [
            {k: (v if k == "date" else float(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]
    test.write_text(json.dumps(profile))
    return test


@fixture
def sim(test_json) -> Simulation:
    return run(test_json)
