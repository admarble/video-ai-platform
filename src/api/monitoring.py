from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta

from ..monitoring import CostMonitor
from ..monitoring.auto_scaler import AutoScaler
from ..services.video.cost_aware_processor import CostAwareVideoProcessor

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

def get_cost_monitor() -> CostMonitor:
    """Dependency to get cost monitor instance"""
    # This should be properly initialized in your app startup
    raise NotImplementedError("Cost monitor dependency not configured")

def get_auto_scaler() -> AutoScaler:
    """Dependency to get auto scaler instance"""
    # This should be properly initialized in your app startup
    raise NotImplementedError("Auto scaler dependency not configured")

def get_video_processor() -> CostAwareVideoProcessor:
    """Dependency to get video processor instance"""
    # This should be properly initialized in your app startup
    raise NotImplementedError("Video processor dependency not configured")

@router.get("/metrics/current")
async def get_current_metrics(
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
) -> Dict[str, Any]:
    """Get current system metrics including resource utilization and scaling state"""
    try:
        metrics = await auto_scaler._get_current_metrics()
        resources = await auto_scaler._get_resource_utilization()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_instances": auto_scaler.current_instances,
            "metrics": metrics,
            "resources": resources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/history")
async def get_metrics_history(
    days: int = 7,
    cost_monitor: CostMonitor = Depends(get_cost_monitor)
) -> Dict[str, Any]:
    """Get historical metrics and cost data"""
    try:
        cost_summary = cost_monitor.get_cost_summary(days=days)
        daily_metrics = cost_monitor.get_cost_history(days=days)
        
        return {
            "cost_summary": cost_summary,
            "daily_metrics": daily_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scaling/rules")
async def get_scaling_rules(
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
) -> Dict[str, Any]:
    """Get current auto-scaling rules and their states"""
    try:
        rules = {}
        for name, rule in auto_scaler.scaling_rules.items():
            rules[name] = {
                "trigger": rule.trigger.value,
                "threshold": rule.threshold,
                "scale_factor": rule.scale_factor,
                "cooldown_minutes": rule.cooldown_minutes,
                "last_scaled": datetime.fromtimestamp(rule.last_scaled).isoformat() 
                    if rule.last_scaled else None
            }
        
        return {
            "current_instances": auto_scaler.current_instances,
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scaling/history")
async def get_scaling_history(
    days: int = 7,
    cost_monitor: CostMonitor = Depends(get_cost_monitor)
) -> List[Dict[str, Any]]:
    """Get historical scaling events"""
    try:
        # Get scaling events from cost monitor history
        history = cost_monitor.get_cost_history(days=days)
        scaling_events = []
        
        for entry in history:
            if 'scaling_event' in entry:
                scaling_events.append({
                    "timestamp": entry['timestamp'],
                    "previous_instances": entry['scaling_event']['previous'],
                    "new_instances": entry['scaling_event']['new'],
                    "trigger": entry['scaling_event']['trigger'],
                    "reason": entry['scaling_event']['reason']
                })
        
        return scaling_events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing/efficiency")
async def get_processing_efficiency(
    days: int = 30,
    video_processor: CostAwareVideoProcessor = Depends(get_video_processor)
) -> Dict[str, Any]:
    """Get video processing efficiency metrics"""
    try:
        return video_processor.analyze_processing_efficiency(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/costs/summary")
async def get_cost_summary(
    days: int = 30,
    video_id: Optional[str] = None,
    video_processor: CostAwareVideoProcessor = Depends(get_video_processor)
) -> Dict[str, Any]:
    """Get cost summary with optimization impact"""
    try:
        return video_processor.get_processing_costs(
            video_id=video_id,
            days=days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/resources/limits")
async def get_resource_limits(
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
) -> Dict[str, Any]:
    """Get current resource limits configuration"""
    try:
        return {
            "max_cost_per_hour": auto_scaler.resource_limits.max_cost_per_hour,
            "max_instances": auto_scaler.resource_limits.max_instances,
            "min_free_memory": auto_scaler.resource_limits.min_free_memory,
            "max_gpu_memory": auto_scaler.resource_limits.max_gpu_memory,
            "max_storage": auto_scaler.resource_limits.max_storage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 