class ConditionalPDist:

    def __init__(self):
        raise NotImplementedError()

    def prob(self, value, context):
        raise NotImplementedError()

class FreqCPD(ConditionalPDist):

    def __init__(self, cfd):
