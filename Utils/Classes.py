
class Classes():
    (NONSPEECH, SPEECH) = list(range(2)) # 0 to NONSPEECH, 1 to SPEECH

    def getLabel(self, i):
        # convert integer into label
        if i > 1:
            raise Exception("Integer out of range: no class with number ",i)
        if i == 0:
            return self.NONSPEECH
        return self.SPEECH
