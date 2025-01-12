from rate_limiter import (
    setup_rate_limiter,
    RateLimitType,
    RateLimitDecorator,
    RateLimitExceeded
)
import time
from typing import Dict, Any

# Initialize rate limiter
rate_limiter = setup_rate_limiter(redis_url="redis://localhost:6379")

# Apply rate limiting to video processing
@RateLimitDecorator(
    rate_limiter,
    RateLimitType.PROCESSING,
    scope_func=lambda user_id, **kwargs: f"user:{user_id}"
)
def process_video(user_id: str, video_path: str) -> Dict[str, Any]:
    # Simulate video processing
    time.sleep(2)
    return {
        "status": "success",
        "user_id": user_id,
        "video_path": video_path,
        "processed_at": time.time()
    }

# Apply rate limiting to uploads
@RateLimitDecorator(
    rate_limiter,
    RateLimitType.UPLOADS,
    scope_func=lambda user_id, **kwargs: f"user:{user_id}"
)
def upload_video(user_id: str, video_file: str) -> str:
    # Simulate video upload
    time.sleep(1)
    return f"uploads/{user_id}/{video_file}"

def main():
    user_id = "user123"
    
    try:
        # Try to process multiple videos
        for i in range(15):  # This should exceed the processing limit
            try:
                result = process_video(user_id, f"video_{i}.mp4")
                print(f"Processed video {i}: {result}")
            except RateLimitExceeded as e:
                print(f"Rate limit exceeded on video {i}: {str(e)}")
                break
            
        # Try to upload multiple videos
        for i in range(25):  # This should exceed the upload limit
            try:
                path = upload_video(user_id, f"video_{i}.mp4")
                print(f"Uploaded video {i} to {path}")
            except RateLimitExceeded as e:
                print(f"Rate limit exceeded on upload {i}: {str(e)}")
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 