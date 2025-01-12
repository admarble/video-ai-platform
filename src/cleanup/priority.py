from enum import Enum

class CleanupPriority(Enum):
    """Cleanup task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        priority_order = {
            CleanupPriority.LOW: 0,
            CleanupPriority.MEDIUM: 1,
            CleanupPriority.HIGH: 2,
            CleanupPriority.CRITICAL: 3
        }
        return priority_order[self] < priority_order[other] 