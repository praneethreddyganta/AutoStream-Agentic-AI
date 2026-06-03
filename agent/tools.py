def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock lead capture API function.
    Fires ONLY when name, email, and platform are fully collected.
    """
    result = f"Lead captured successfully: {name}, {email}, {platform}"
    print(f"\n[TOOL EXECUTION] {result}\n")
    return result
