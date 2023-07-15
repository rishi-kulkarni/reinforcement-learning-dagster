import numpy as np

# Define constants
BASE_MARGIN = 30
COUPON_USERS_MARGIN = 25
COUPON_EFFECT_OLD_ANDROID = [0, 7, 15, 15]
COUPON_EFFECT_NEW = [0, 3, 5, 5]
GEO_BASE_PROB = {
    "Chicago": 0.1,
    "Indianapolis": 0.1,
    "Albuquerque": 0.1,
    "New York City": 0.15,
    "Bay Area": 0.15,
    "Greater Boston": 0.15,
}
NOISE_STD_DEV = 3.0


def base_margin(geo: str, device: str, coupon: int) -> float:
    # Calculate base probability
    base_prob = GEO_BASE_PROB[geo]

    # Calculate coupon effect
    if device in ["Old iPhone", "Android"]:
        coupon_effect = COUPON_EFFECT_OLD_ANDROID[coupon // 5] / 100
    else:  # New iPhone
        coupon_effect = COUPON_EFFECT_NEW[coupon // 5] / 100

    # Determine if the user would have converted without a coupon,
    # if they're an incremental customer, or if they fail to convert
    outcome = np.random.choice(
        ["base", "incremental", "fail"],
        p=[base_prob, coupon_effect, 1 - base_prob - coupon_effect],
    )

    # Calculate margin
    if outcome == "base":
        return BASE_MARGIN - coupon
    elif outcome == "incremental":
        return COUPON_USERS_MARGIN - coupon
    else:
        return 0


def simulated_margin(geo: str, device: str, coupon: int) -> float:
    # Add noise to the expected margin for users who convert
    noise = np.random.normal(0, NOISE_STD_DEV)
    margin = base_margin(geo, device, coupon)
    if margin != 0:
        return margin + noise
    else:
        return 0


def expected_margin(geo: str, device: str, coupon: int) -> float:
    # Calculate base probability
    base_prob = GEO_BASE_PROB[geo]

    # Calculate coupon effect
    if device in ["Old iPhone", "Android"]:
        coupon_effect = COUPON_EFFECT_OLD_ANDROID[coupon // 5] / 100
    else:  # New iPhone
        coupon_effect = COUPON_EFFECT_NEW[coupon // 5] / 100

    expectation = base_prob * (BASE_MARGIN - coupon) + coupon_effect * (
        COUPON_USERS_MARGIN - coupon
    )

    return expectation
