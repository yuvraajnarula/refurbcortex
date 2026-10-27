from app.core.cost_estimator import CostEstimator


def test_cost_estimator_deterministic():
    c = CostEstimator()
    b1 = c.estimate(panel="front_bumper", severity="MODERATE", area_cm2=100.0,
                    city_tier="tier_2", vehicle_age=5, odometer_km=50000, brand="Maruti")
    b2 = c.estimate(panel="front_bumper", severity="MODERATE", area_cm2=100.0,
                    city_tier="tier_2", vehicle_age=5, odometer_km=50000, brand="Maruti")
    assert b1.total_estimated_inr == b2.total_estimated_inr
    assert b1.cost_range_inr[0] < b1.cost_range_inr[1]


def test_cost_estimator_varies_with_area():
    c = CostEstimator()
    small = c.estimate(panel="front_bumper", severity="MODERATE", area_cm2=10.0,
                       city_tier="tier_2", vehicle_age=5, odometer_km=50000, brand="Maruti")
    large = c.estimate(panel="front_bumper", severity="MODERATE", area_cm2=1000.0,
                       city_tier="tier_2", vehicle_age=5, odometer_km=50000, brand="Maruti")
    # Paint cost scales with area for MODERATE (needs_paint=True)
    assert large.total_estimated_inr > small.total_estimated_inr


def test_cost_estimator_bounded():
    c = CostEstimator()
    b = c.estimate(panel="unknown_panel", severity="CRITICAL", area_cm2=1.0,
                   city_tier="tier_1", vehicle_age=20, odometer_km=300000, brand="Ferrari")
    assert b.total_estimated_inr > 0
    assert b.cost_range_inr[0] > 0
    assert b.cost_range_inr[1] > b.cost_range_inr[0]
