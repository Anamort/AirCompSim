import time
from enum import Enum
from Application import Task
from Location import Location
from typing import Optional

class EventType(Enum):
    Offload = 1
    Process = 2
    Returned = 3
    UserMove = 4
    UserStop = 5
    UAVStop = 6
    UAVMove = 7
    State = 8



class Event(object):
    id: int = 0
    events = []

    def __init__(self, type: EventType, task, simTime, uav, user, loc: Location):
        self.id = Event.id
        Event.id += 1
        Event.events.append(self)
        self.scheduledTime = simTime
        self.type = type
        self.fromLayer = None
        self.toLayer = None
        self.task: Task = task
        self.uav = uav
        self.user = user
        self.location = loc

    def __lt__(self, other):
        if self.scheduledTime < other.scheduledTime:
            return True
        return False

    @classmethod
    def resetAll(cls):
        Event.id = 0
        Event.events = []


