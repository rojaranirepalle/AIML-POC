def risk_level(gap):

    if gap > 0.15:
        return "🔴 High Risk"
    elif gap > 0.07:
        return "🟠 Medium Risk"
    return "🟢 Safe"