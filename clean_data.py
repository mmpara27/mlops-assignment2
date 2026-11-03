import numpy as np

def clean_data(data):
    # Remove rows with missing required values.
    data = data.dropna(
        subset=[
            "region",
            "age",
            "weight",
            "height",
            "howlong",
            "gender",
            "eat",
            "train",
            "background",
            "experience",
            "schedule",
            "deadlift",
            "candj",
            "snatch",
            "backsq",
        ]
    )

    # Remove columns we do not need.
    data = data.drop(
        columns=[
            "affiliate",
            "team",
            "name",
            "athlete_id",
            "fran",
            "helen",
            "grace",
            "filthy50",
            "fgonebad",
            "run400",
            "run5k",
            "pullups",
            "train",
        ]
    )

    # Remove outliers.
    data = data[data["weight"] < 1500]
    data = data[data["gender"] != "--"]
    data = data[data["age"] >= 18]
    data = data[(data["height"] < 96) & (data["height"] > 48)]

    # Changed the undefined v2 reference to data. Filter logic is unchanged.
    data = data[
        ((data["deadlift"] > 0) & (data["deadlift"] <= 1105))
        | ((data["gender"] == "Female") & (data["deadlift"] <= 636))
    ]
    data = data[(data["candj"] > 0) & (data["candj"] <= 395)]
    data = data[(data["snatch"] > 0) & (data["snatch"] <= 496)]
    data = data[(data["backsq"] > 0) & (data["backsq"] <= 1069)]

    # Clean survey data.
    decline_dict = {"Decline to answer|": np.nan}
    data = data.replace(decline_dict)
    data = data.dropna(
        subset=["background", "experience", "schedule", "howlong", "eat"]
    )
    return data.copy()
