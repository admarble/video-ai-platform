from typing import List, Dict, Any, Optional
import asyncio
import os
from pathlib import Path
import ffmpeg
from ..models import VideoScene, TimeSegment

class ClipGenerator:
    def __init__(
        self,
        temp_dir: str,
        max_concurrent: int = 5
    ):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def create_clips(
        self,
        scenes: List[VideoScene],
        video_id: str
    ) -> List[Dict[str, Any]]:
        """Generate clips for multiple scenes"""
        clips = []
        for scene in scenes:
            clip = await self.generate_clip(
                video_id=video_id,
                start_time=scene.segment.start_time,
                end_time=scene.segment.end_time,
                scene_id=scene.embedding_id
            )
            clips.append(clip)
        return clips
        
    async def generate_clip(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        scene_id: str,
        format: str = 'mp4'
    ) -> Dict[str, Any]:
        """Generate a video clip for a specific scene"""
        async with self.semaphore:
            try:
                # Generate output path
                output_path = self.temp_dir / f"{video_id}_{scene_id}.{format}"
                
                # Get video path from video ID
                video_path = await self._get_video_path(video_id)
                
                # Generate clip using ffmpeg
                await self._generate_clip_ffmpeg(
                    input_path=video_path,
                    output_path=output_path,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Get clip duration
                duration = end_time - start_time
                
                return {
                    'url': str(output_path),
                    'duration': duration,
                    'format': format,
                    'scene_id': scene_id,
                    'video_id': video_id,
                    'timeframe': {
                        'start': start_time,
                        'end': end_time
                    }
                }
                
            except Exception as e:
                raise Exception(f"Error generating clip: {str(e)}")
                
    async def _generate_clip_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float
    ):
        """Generate clip using ffmpeg"""
        try:
            # Build ffmpeg command
            stream = ffmpeg.input(
                input_path,
                ss=start_time,
                t=end_time - start_time
            )
            
            # Add video stream
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec='aac',
                vcodec='h264',
                preset='fast',
                movflags='faststart',
                loglevel='error'
            )
            
            # Run ffmpeg
            await asyncio.create_subprocess_exec(
                *stream.compile(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg error: {e.stderr.decode()}")
            
    async def _get_video_path(self, video_id: str) -> str:
        """Get video file path from video ID"""
        # This would be replaced with actual video storage logic
        # For now, assume video ID is the path
        if not os.path.exists(video_id):
            raise Exception(f"Video not found: {video_id}")
        return video_id
        
    async def cleanup_clip(self, clip_path: str):
        """Clean up generated clip file"""
        try:
            os.remove(clip_path)
        except Exception as e:
            print(f"Error cleaning up clip {clip_path}: {str(e)}")
            
    async def cleanup_all(self):
        """Clean up all generated clips"""
        try:
            for file in self.temp_dir.glob("*"):
                await self.cleanup_clip(str(file))
        except Exception as e:
            print(f"Error cleaning up clips: {str(e)}")
