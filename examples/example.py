import asyncio
import logging
from src.core.circuit_breaker import CircuitBreakerRegistry, CircuitConfig, CircuitOpenError, circuit_breaker

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create a registry
registry = CircuitBreakerRegistry()

# Create a custom configuration
config = CircuitConfig(
    failure_threshold=3,        # Open after 3 failures
    reset_timeout=30,          # Try to reset after 30 seconds
    half_open_limit=2,         # Allow 2 test requests when half-open
    window_size=60,            # Count failures in 60 second window
    success_threshold=2        # Close after 2 consecutive successes
)

# Example async function with circuit breaker
@circuit_breaker("example_service", registry, config)
async def example_service(succeed: bool = True):
    """Example service that can succeed or fail on demand"""
    if not succeed:
        raise Exception("Service failed!")
    return "Service succeeded!"

# Fallback function for when circuit is open
async def fallback_function(*args, **kwargs):
    return "Using fallback function"

# Example with fallback
@circuit_breaker("example_with_fallback", registry, config, fallback=fallback_function)
async def service_with_fallback(succeed: bool = True):
    """Example service with fallback behavior"""
    if not succeed:
        raise Exception("Service failed!")
    return "Service succeeded!"

async def main():
    # Test normal operation
    try:
        result = await example_service(succeed=True)
        print("Success:", result)
    except Exception as e:
        print("Error:", str(e))

    # Test multiple failures
    for i in range(4):
        try:
            result = await example_service(succeed=False)
            print("Success:", result)
        except CircuitOpenError as e:
            print("Circuit open:", str(e))
        except Exception as e:
            print("Error:", str(e))

    # Test fallback behavior
    for i in range(4):
        result = await service_with_fallback(succeed=False)
        print("Fallback result:", result)

    # Print circuit states
    print("\nCircuit States:")
    for name, state in registry.get_all_states().items():
        print(f"{name}: {state}")

if __name__ == "__main__":
    asyncio.run(main()) 