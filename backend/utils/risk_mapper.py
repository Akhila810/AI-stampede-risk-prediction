def risk_to_color(risk_class):
    if risk_class == 2:
        return "red"
    elif risk_class == 1:
        return "yellow"
    else:
        return "green"
