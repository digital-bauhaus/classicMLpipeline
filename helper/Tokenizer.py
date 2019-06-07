from subprocess import Popen, PIPE

# tokenizer needs this tool installed: https://github.com/dspinellis/tokenizer
class Tokenizer(object):
    def __init__(self):
        pass

    def __call__(self, text):
        proc = Popen(["/usr/local/bin/tokenizer", "-l", "Java", "-t", "c"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = proc.communicate(str.encode(text))
        tokens = output.decode().split("\n")[:-1]
        return tokens