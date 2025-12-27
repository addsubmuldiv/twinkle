from .base import Preprocessor
from ..trajectory import Trajectory, Message


class CompetitionMathProcessor(Preprocessor):

    def __call__(self, row):
        problem = row['problem']
        solution = row['solution']
        messages = [
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)
