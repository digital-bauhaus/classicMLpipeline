from ..Feature import Feature


class NumOfLinesUntilCurrentStatement(Feature):
    FEATURE_NAME = "NumOfLinesUntilCurrentStatement"

    def __init__(self):
        super(NumOfLinesUntilCurrentStatement, self).__init__()
