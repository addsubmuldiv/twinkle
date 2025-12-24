from .base import DataProcessor
from ..trajectory import Trajectory, Message


class CompetitionMathProcessor(DataProcessor):

    def __call__(self, rows):
        trajectories = []
        for row in rows:
            problem = row['problem']
            solution = row['solution']
            messages = [
                Message(role='user', content=problem),
                Message(role='assistant', content=solution),
            ]
            trajectories.append(Trajectory(messages=messages))
        return trajectories
