import argparse
import itertools
import math
import operator
import re
import sys

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        if args:
            print(*args, sep = '\n', file = sys.stderr)
            sys.exit(1)
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, attr):
        try:
            ret = super().__getattribute__(attr)
            has = True
        except AttributeError:
            has = False
            
        if not has:
            return False
        return ret

class Stack:
    def __init__(self, array = None):
        self.elements = list(array if array is not None else [])

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return str(self.elements)
        
    def push(self, *values):
        for value in values:
            self.elements.append(value)
        if values:
            return value

    def peek(self, index = -1):
        try:
            return self.elements[index]
        except:
            return 0

    def pop(self):
        try:
            return self.elements.pop()
        except:
            return 0
    
def convert(opcodes, modifier = 0):
    opcodes = list(map(lambda a: bin(a)[2:].zfill(5), opcodes))
    bytecodes = ''.join(opcodes)
    mod = bin(modifier)[2:]
    
    while len(bytecodes + mod) % 8:
        mod = '0' + mod
    bytecodes += mod

    binbytes = []
    for index, bit in enumerate(bytecodes):
        if not(index % 8):
            binbytes.append('')
        binbytes[-1] += bit
        
    return ' '.join(map(lambda a: hex(int(a, 2))[2:], binbytes))

def execute(code, samples, inputted, flags):
    if 'a' not in flags:
        inputted = list(map(ord, inputted))
    else:
        inputted = list(inputted)

    opcodes, mod = code

    if tuple(inputted) in samples.keys():
        return samples[tuple(inputted)]

    if len(samples.keys()) == 0:
        toggle = 1
        for i in getpaths(opcodes):
            path = i
            break

    elif '_' in samples.keys():
        return samples['_']

    else:
        possible_paths = []
        flatpaths = []

        for samplein, sampleout in samples.items():
            outpaths = []
            for path in getpaths(opcodes):
                stack = Stack(samplein)
                stack = run_path(path, stack)

                if 's' in flags:
                    out = stack.elements
                else:
                    out = stack.pop()

                if 'a' in flags:
                    out = make_ascii(out)

                if out == sampleout:
                    outpaths.append(path)
            possible_paths.append(outpaths)
            flatpaths.extend(outpaths)

        unique = []
        for elem in flatpaths:
            if elem not in unique:
                unique.append(elem)

        matches = []
        for path in unique:
            if all(path in pos for pos in possible_paths):
                matches.append(path)

        if len(matches) != 1:
            raise Exception('Bad sample I/Os; unable to find unique path')
        
        path = matches[0]
        
    stack = Stack(inputted)
    ret = run_path(path, stack)

    if 's' in flags: ret = ret.elements
    else: ret = ret.pop()
    if 'a' in flags: ret = make_ascii(ret)

    return ret

def factor(x):
    for i in range(1, x + 1):
        if not (x % i):
            yield i

def frombase(digits, base):
    total = 0
    for index, elem in enumerate(digits[::-1]):
        total += elem * base ** index
    return total

def getpaths(codes):
    cmdlists = []
    for code in codes:
        temp = []
        for cmd in COMMANDS[code].cmds:
            temp.append(AttrDict(
                arity = COMMANDS[code].arity,
                vec = COMMANDS[code].vec,
                cmd = cmd
            ))
        cmdlists.append(temp)
    return itertools.product(*cmdlists)

def get_samples(filename, read_file = True):
    if read_file:
        with open(filename) as file:
            lines = file.readlines()
    else:
        lines = filename.splitlines()
    mapping = {}
        
    for line in lines:
        ins, out = re.split(r'[^\d.\s]', line, maxsplit = 1)
        if ' ' in ins:
            ins = ins.split()
        else:
            ins = [ins]
        ins = tuple(map(lambda a: eval(a) if a != '_' else a, ins))
        mapping[ins] = eval(out)

    return mapping

def isfac(x):
    return x in list(map(math.factorial, range(x + 1)))

def isfib(x):
    a, b = 1, 1
    while b < x:
        a, b = b, a + b
    return b == x

def isneg(x):
    return x < 0

def ispos(x):
    return x > 0

def isprime(x):
    return len(list(factor(x))) == 2

def lcm(x, y):
    return x * y // math.gcd(x, y)

def make_ascii(arg):
    if isinstance(arg, list):
        return ''.join(map(chr, filter(lambda a: isinstance(a, int) and -1 < a < 0x110000, arg)))
    return chr(arg)

def nthfib(x):
    a, b = 1, 1
    for _ in range(x - 1):
        a, b = b, a + b
    return a

def nthprime(x):
    i = 1
    while x:
        i += 1
        x -= isprime(i)
    return i

def read_prog(filename):
    with open(filename, mode = 'rb') as file:
        contents = file.read()
        
    bytecode = ''
    for char in contents:
        bytecode += bin(char)[2:].zfill(8)
        
    bincode = []
    for index, bit in enumerate(bytecode):
        if not(index % 5):
            bincode.append('')
        bincode[-1] += bit

    modifer = 0
    if len(bincode[-1]) != 5:
        modifer = int(bincode.pop(), 2)

    opcodes = list(map(lambda a: int(a, 2), bincode))

    return (opcodes, modifer)

def run(op, *args):
    try:
        ret = op.cmd(*args)
        if hasattr(ret, '__iter__'):
            ret = list(ret)
    except:
        ret = 0
    return ret

def run_path(path, stack):
    maxiters = 1 << 32
    index = looplevel = 0
    start_indexes = []
    
    while index < len(path):
        op = path[index]
        
        if op.arity == -1:

            if op.cmd == 'dup':
                a = stack.pop()
                stack.push(a, a)
                
            if op.cmd == 'open loop':
                if stack.peek():
                    looplevel += 1
                    start_indexes.append(index)

                else:
                    newindex = 0
                    count = -1
                    for i, op in enumerate(path[::-1]):
                        if op.cmd == 'close loop':
                            count += 1
                        if count == looplevel:
                            newindex = len(path) - i
                            break
                    if newindex < index:
                        raise Exception('Unclosed loop')
                    index = newindex

            if op.cmd == 'close loop':
                if stack.peek() and maxiters:
                    index = start_indexes[-1]
                    maxiters -= 1

                else:
                    start_indexes.pop()
                
        else:
            args = [stack.pop() for _ in range(op.arity)]
            if op.vec and any(hasattr(a, '__iter__')for a in args) and args:
                if op.arity == 1:
                    ret = [run(op, arg) for arg in args]
                    
                if op.arity == 2:
                    l, r = args
                    if hasattr(l, '__iter__'):
                        if hasattr(r, '__iter__'):
                            ret = [run(op, a, b) for a in l for b in r]
                        else:
                            ret = [run(op, a, r) for a in l]
                    else:
                        ret = [run(op, l, a) for a in r]

            else:
                ret = run(op, *args)

            if hasattr(ret, '__iter__'):
                stack.push(*ret)
            if ret is not None:
                stack.push(ret)

        index += 1
            
    return stack

def stdin_proc(contents, samples_in_file):
    if samples_in_file:
        return contents
    else:
        inputs, samples = contents.split('\n' + '=' * 10 + '\n')
        samples = get_samples(samples, read_file = False)
        return inputs, samples
        

def tobase(integer, base):
    digits = []
    sign = (integer > 0) - (integer < 0)
    integer = abs(integer)
    
    while integer:
        integer, rem = divmod(integer, base)
        digits.append(rem)
        
    return list(map(lambda a: sign * a, digits[::-1]))

COMMANDS = {

    0x00: AttrDict(
        arity = 2,
        vec = False,
        cmds = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.floordiv,
            math.log,
            divmod,
        ]
    ),

    0x01: AttrDict(
        arity = 2,
        vec = False,
        cmds = [
            operator.and_,
            operator.or_,
            operator.xor,
            lambda a, b: ~(a & b),
            lambda a, b: ~(a | b),
            lambda a, b: ~(a ^ b),
            operator.lshift,
            operator.rshift,
        ]
    ),

    0x02: AttrDict(
        arity = 2,
        vec = False,
        cmds = [
            operator.eq,
            operator.ne,
            operator.lt,
            operator.gt,
            operator.le,
            operator.ge,
            lambda a, b: not(a < b),
            lambda a, b: not(a > b),
            lambda a, b: not(a <= b),
            lambda a, b: not(a >= b),
        ]
    ),

    0x03: AttrDict(
        arity = 2,
        vec = False,
        cmds = [
            tobase,
            frombase,
            math.gcd,
            lcm,
            min,
            max,
            range,
            math.hypot,
            math.atan2,
        ]
    ),

    0x04: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            math.sin,
            math.cos,
            math.tan,
            lambda a: 1 / math.cos(a),
            lambda a: 1 / math.sin(a),
            lambda a: 1 / math.tan(a),
        ]
    ),

    0x05: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            math.asin,
            math.acos,
            math.atan,
            lambda a: math.acos(1 / a),
            lambda a: math.asin(1 / a),
            lambda a: math.atan(1 / a),
        ]
    ),

    0x06: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            math.sinh,
            math.cosh,
            math.tanh,
            lambda a: 1 / math.cosh(a),
            lambda a: 1 / math.sinh(a),
            lambda a: 1 / math.tanh(a),
        ]
    ),

    0x07: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            math.asinh,
            math.acosh,
            math.atanh,
            lambda a: math.acosh(1 / a),
            lambda a: math.asinh(1 / a),
            lambda a: math.atanh(1 / a),
        ]
    ),

    0x08: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            operator.neg,
            operator.inv,
            operator.not_,
            lambda a: tobase(a, 2),
            lambda a: tobase(a, 16),
            lambda a: tobase(a, 10),
            lambda a: tobase(a, 8),
        ]
    ),

    0x09: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            math.sqrt,
            math.factorial,
            lambda a: a ** 2,
            abs,
            math.ceil,
            math.floor,
            lambda a: math.copysign(1, a),
            factor,
            range,
        ]
    ),

    0x0a: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            lambda a: a & 1,
            lambda a: a | 1,
            lambda a: a ^ 1,
            lambda a: a * 0,
            lambda a: a / 2,
            lambda a: a + 1,
            lambda a: a - 1,
            lambda a: a // 1,
            lambda a: 1 / a,
        ]
    ),

    0x0b: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            lambda a: 10*a,
            lambda a: 10*a + 1,
            lambda a: 10*a + 2,
            lambda a: 10*a + 3,
            lambda a: 10*a + 4,
            lambda a: 10*a + 5,
            lambda a: 10*a + 6,
            lambda a: 10*a + 7,
            lambda a: 10*a + 8,
            lambda a: 10*a + 9,
        ]
    ),

    0x0c: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            isprime,
            isfib,
            isfac,
            ispos,
            isneg,
            nthprime,
            nthfib,
        ]
    ),

    0x0d: AttrDict(
        arity = 0,
        vec = False,
        cmds = [
            lambda: 0,
            lambda: 1,
            lambda: 2,
            lambda: 3,
            lambda: 4,
            lambda: 5,
            lambda: 6,
            lambda: 7,
            lambda: 8,
            lambda: 9,
        ]
    ),

    0x0e: AttrDict(
        arity = -1,
        cmds = ['dup']
    ),

    0x0f: AttrDict(
        arity = -1,
        cmds = ['open loop']
    ),

    0x10: AttrDict(
        arity = 2,
        vec = True,
        cmds = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.floordiv,
            math.log,
            divmod,
        ]
    ),

    0x11: AttrDict(
        arity = 2,
        vec = True,
        cmds = [
            operator.and_,
            operator.or_,
            operator.xor,
            lambda a, b: ~(a & b),
            lambda a, b: ~(a | b),
            lambda a, b: ~(a ^ b),
            operator.lshift,
            operator.rshift,
        ]
    ),

    0x12: AttrDict(
        arity = 2,
        vec = True,
        cmds = [
            operator.eq,
            operator.ne,
            operator.lt,
            operator.gt,
            operator.le,
            operator.ge,
            lambda a, b: not(a < b),
            lambda a, b: not(a > b),
            lambda a, b: not(a <= b),
            lambda a, b: not(a >= b),
        ]
    ),

    0x13: AttrDict(
        arity = 2,
        vec = True,
        cmds = [
            tobase,
            frombase,
            math.gcd,
            lcm,
            min,
            max,
            range,
            math.hypot,
            math.atan2,
        ]
    ),

    0x14: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            math.sin,
            math.cos,
            math.tan,
            lambda a: 1 / math.cos(a),
            lambda a: 1 / math.sin(a),
            lambda a: 1 / math.tan(a),
        ]
    ),

    0x15: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            math.asin,
            math.acos,
            math.atan,
            lambda a: math.acos(1 / a),
            lambda a: math.asin(1 / a),
            lambda a: math.atan(1 / a),
        ]
    ),

    0x16: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            math.sinh,
            math.cosh,
            math.tanh,
            lambda a: 1 / math.cosh(a),
            lambda a: 1 / math.sinh(a),
            lambda a: 1 / math.tanh(a),
        ]
    ),

    0x17: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            math.asinh,
            math.acosh,
            math.atanh,
            lambda a: math.acosh(1 / a),
            lambda a: math.asinh(1 / a),
            lambda a: math.atanh(1 / a),
        ]
    ),

    0x18: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            operator.neg,
            operator.inv,
            operator.not_,
            lambda a: tobase(a, 2),
            lambda a: tobase(a, 16),
            lambda a: tobase(a, 10),
            lambda a: tobase(a, 8),
        ]
    ),

    0x19: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            math.sqrt,
            math.factorial,
            lambda a: a ** 2,
            abs,
            math.ceil,
            math.floor,
            lambda a: math.copysign(1, a),
            factor,
            range,
        ]
    ),

    0x1a: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            lambda a: a & 1,
            lambda a: a | 1,
            lambda a: a ^ 1,
            lambda a: a * 0,
            lambda a: a / 2,
            lambda a: a + 1,
            lambda a: a - 1,
            lambda a: a // 1,
            lambda a: 1 / a,
        ]
    ),

    0x1b: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            lambda a: 10*a,
            lambda a: 10*a + 1,
            lambda a: 10*a + 2,
            lambda a: 10*a + 3,
            lambda a: 10*a + 4,
            lambda a: 10*a + 5,
            lambda a: 10*a + 6,
            lambda a: 10*a + 7,
            lambda a: 10*a + 8,
            lambda a: 10*a + 9,
        ]
    ),

    0x1c: AttrDict(
        arity = 1,
        vec = True,
        cmds = [
            isprime,
            isfib,
            isfac,
            ispos,
            isneg,
            nthprime,
            nthfib,
        ]
    ),

    0x1d: AttrDict(
        arity = 0,
        vec = True,
        cmds = [
            lambda: 0,
            lambda: 1,
            lambda: 2,
            lambda: 3,
            lambda: 4,
            lambda: 5,
            lambda: 6,
            lambda: 7,
            lambda: 8,
            lambda: 9,
        ]
    ),

    0x1e: AttrDict(
        arity = 1,
        vec = False,
        cmds = [
            lambda a: None,
        ]
    ),

    0x1f: AttrDict(
        arity = -1,
        cmds = ['close loop']
    ),

}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = './sample')
    
    parser.add_argument('-a', help = 'ASCII I/O', action = 'store_true')
    parser.add_argument('-s', help = 'Stack is final output', action = 'store_true')
    parser.add_argument('-f', help = 'Read samples from file', action = 'store_true')

    parser.add_argument('file')
    parser.add_argument('samples', nargs = '?')
    settings = parser.parse_args()

    flags = list(filter(None, map(lambda a: a[0] if a[1] == True else '', settings._get_kwargs())))
    if 'f' in flags:
        settings.samples = get_samples(settings.samples)
        # stdin = stdin_proc(sys.stdin.read(), True)
        stdin = stdin_proc('', True)
    else:
        settings.samples, stdin = stdin_proc(sys.stdin.read(), False)

    out = execute(read_prog(settings.file), settings.samples, stdin, settings)
    print(out)










        
