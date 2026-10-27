from app.services.web_agent_async import _extract_prices, _normalize_result


def test_extract_prices():
    text = "Maruti Swift front bumper dent repair cost ₹12,000 to ₹18,000 OEM aftermarket labor Rs. 450 per hour"
    prices = _extract_prices(text)
    assert 12000.0 in prices
    assert 18000.0 in prices
    assert 450.0 not in prices  # sub-500 filtered out


def test_normalize_result_fills_defaults():
    data = _normalize_result("Maruti", "front_bumper", "dent", [])
    assert data["source"] == "duckduckgo"
    assert data["brand"] == "Maruti"
    assert data["panel"] == "front_bumper"
    assert data["category"] == "dent"
    assert data["oem_part_price"] == 12000.0
    assert data["aftermarket_price"] == 6500.0
    assert data["recall"] == "NONE"
