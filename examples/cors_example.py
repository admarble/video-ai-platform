from pathlib import Path
from typing import List
from src.cors_config import CORSConfig, CORSMode, CORSRule

def example_cors_usage():
    # Example 1: Create with default strict mode
    cors_config = CORSConfig(mode=CORSMode.STRICT)

    # Example 2: Create with custom config file
    config_path = Path("src/cors_config.json")
    cors_config = CORSConfig(
        mode=CORSMode.CUSTOM,
        config_path=config_path
    )

    # Example 3: Add a custom rule programmatically
    cors_config.add_rule("api", CORSRule(
        allowed_origins=["https://app.example.com", "https://*.example.com"],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
        allowed_headers=["Content-Type", "Authorization", "X-Request-ID"],
        expose_headers=["Content-Length"],
        max_age=3600,
        allow_credentials=True
    ))

def process_request(
    cors_config: CORSConfig,
    origin: str,
    method: str,
    headers: List[str]
):
    # Example request processing with CORS
    
    # Check if origin is allowed
    if not cors_config.is_origin_allowed(origin, "api"):
        return {"error": "Origin not allowed"}, 403
        
    # Get CORS headers
    cors_headers = cors_config.get_cors_headers(
        origin=origin,
        request_method=method,
        request_headers=headers,
        rule_name="api"
    )
    
    # Handle preflight requests
    if method == "OPTIONS":
        return {}, 200, cors_headers
        
    # Process regular request (example)
    response = {"message": "Request processed successfully"}
    return response, 200, cors_headers

if __name__ == "__main__":
    # Example usage
    cors_config = CORSConfig(mode=CORSMode.CUSTOM, config_path=Path("src/cors_config.json"))
    
    # Example request
    origin = "https://app.example.com"
    method = "POST"
    headers = ["Content-Type", "Authorization"]
    
    response, status, cors_headers = process_request(cors_config, origin, method, headers)
    print(f"Response: {response}")
    print(f"Status: {status}")
    print(f"CORS Headers: {cors_headers}") 