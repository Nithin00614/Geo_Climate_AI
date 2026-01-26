def compute_risk(entry):
    temp = entry["temperature"]
    hum = entry["humidity"]
    aqi = entry["aqi"]

    risk = (
        (temp - 25) * 0.4 +
        (60 - hum) * 0.25 +
        (aqi / 20) * 0.35
    )

    risk = max(risk, 0)

    if risk > 30:
        sev = "High"
    elif risk > 15:
        sev = "Medium"
    else:
        sev = "Low"

    return risk, sev
