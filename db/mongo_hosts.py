def normalize_mongo_hosts(value):
    """Trim seed-list whitespace without rewriting MongoDB connection URIs."""
    value = value.strip()
    if value.startswith(("mongodb://", "mongodb+srv://")):
        return value
    return ",".join(host.strip() for host in value.split(",") if host.strip())
