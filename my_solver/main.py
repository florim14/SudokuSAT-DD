import signal
import os
import time
import sys
import logging
import psutil

import numpy as np
import math
import optparse
import re


def handler(signum, frame):
    logging.error('signum %s' % signum)
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        logging.error('Child pid is {}\n'.format(child.pid))
        logging.error('Killing child.')
        try:
            os.kill(child.pid, 15)
        except OSError as e:
            logging.warn('Process might already be gone. See error below.')
            logging.warn('%s' % str(e))

    logging.warn('SIGNAL received')
    if signum == 15:
        raise TimeoutException('signal')
    else:
        raise KeyboardInterrupt('signal')

def nothing(signum, frame):
    logging.warn('SIGNAL received\n')
    logging.warn('SIGNAL ignored...\n')

def testUnit(clause1, clause2, unitClauses):
    inOrOut = np.any(unitClauses[:] == clause1)
    if not inOrOut:
        inOrOut = np.any(unitClauses[:] == clause2)
        if not inOrOut:
            inOrOut = np.any(unitClauses[:] == -clause1)
            if inOrOut:
                unitClauses = np.append(unitClauses, clause2)
                return False, unitClauses
            inOrOut = np.any(unitClauses[:] == -clause2)
            if inOrOut:
                unitClauses = np.append(unitClauses, clause1)
            return False, unitClauses
    return True, unitClauses

def test(gridSize, blockSize, unitClauses):
    variables = gridSize * gridSize * gridSize
    startClauses = 0
    cnf = ""

    sudo_size = gridSize + 1

    # every cell must contain a value
    for i in range(1, sudo_size):
        for j in range(1, sudo_size):
            clause = ""
            for k in range(1, sudo_size):
                literal = int(i * sudo_size * sudo_size + j * sudo_size + k)
                inOrOut = np.any(unitClauses[:] == literal)
                if inOrOut:
                    break
                clause += str(literal) + " "
                if k == gridSize:
                    cnf += clause
                    cnf += "0\n"
                    startClauses += 1

    # if a row i and column j contains value k, then k should not reappear in the same column
    for y in range(1, sudo_size):
        for v in range(1, sudo_size):
            for x in range(1, sudo_size - 1):
                for w in range(x + 1, sudo_size):
                    first = -(x * sudo_size * sudo_size + y * sudo_size + v)
                    second = -(w * sudo_size * sudo_size + y * sudo_size + v)
                    unit, unitClauses = testUnit(first, second, unitClauses)
                    if not unit:
                        cnf += str(first) + " " + str(second) + " 0\n"
                        startClauses += 1

    # if a row i and column j contains value k, then k should not reappear in the same row
    for x in range(1, sudo_size):
        for v in range(1, sudo_size):
            for y in range(1, sudo_size - 1):
                for w in range(y + 1, sudo_size):
                    first = -(x * sudo_size * sudo_size + y * sudo_size + v)
                    second = -(x * sudo_size * sudo_size + w * sudo_size + v)
                    unit, unitClauses = testUnit(first, second, unitClauses)
                    if not unit:
                        cnf += str(first) + " " + str(second) + " 0\n"
                        startClauses += 1

    # each cell can contain at must one value
    for x in range(1, sudo_size):
        for y in range(1, sudo_size):
            for v in range(1, sudo_size - 1):
                for w in range(v + 1, sudo_size):
                    first = -(x * sudo_size * sudo_size + y * sudo_size + v)
                    second = -(x * sudo_size * sudo_size + y * sudo_size + w)
                    unit, unitClauses = testUnit(first, second, unitClauses)
                    if not unit:
                        cnf += str(first) + " " + str(second) + " 0\n"
                        startClauses += 1
    
    # uniqueness for blocks
    for z in range(1, sudo_size):
        for i in range(0, blockSize):
            for j in range(0, blockSize):
                for x in range(1, blockSize + 1):
                    for y in range(1, blockSize + 1):
                        for k in range(y + 1, blockSize + 1):
                            first = -((blockSize * i + x) * sudo_size * sudo_size + (blockSize * j + y) * sudo_size + z)
                            second = -((blockSize * i + x) * sudo_size * sudo_size + (blockSize * j + k) * sudo_size + z)
                            unit, unitClauses = testUnit(first, second, unitClauses)
                            if not unit:
                                cnf += str(first) + " " + str(second) + " 0\n"
                                startClauses += 1

    for z in range(1, sudo_size):
        for i in range(0, blockSize):
            for j in range(0, blockSize):
                for x in range(1, blockSize + 1):
                    for y in range(1, blockSize + 1):
                        for k in range(x + 1, blockSize + 1):
                            for l in range(1, blockSize + 1):
                                first = -((blockSize * i + x) * sudo_size * sudo_size + (blockSize * j + y) * sudo_size + z)
                                second = -((blockSize * i + k) * sudo_size * sudo_size + (blockSize * j + l) * sudo_size + z)
                                unit, unitClauses = testUnit(first, second, unitClauses)
                                if not unit:
                                    cnf += str(first) + " " + str(second) + " 0\n"
                                    startClauses += 1

    return startClauses, cnf, unitClauses

def read_a_file(file_name):
    f = open(file_name, "r")
    f1 = f.readlines()
    row = 1
    column = 1
    sudo_size = 0
    cnf = ""
    clauses = 0
    unitClauses = np.array([])
    sudokuSize = 0
    for x in f1:
        if x.startswith("puzzle size: "):
            sudokuSize = x.split("x")[1]
            sudo_size = int(sudokuSize) + 1
        if x.startswith("|"):
            x = x.split(" ")
            for letter in x:
                if letter.isnumeric():
                    unit = row * sudo_size * sudo_size + column * sudo_size + int(letter)
                    cnf += str(unit) + " 0\n"
                    unitClauses = np.append(unitClauses, unit)
                    clauses += 1
                    column += 1
                elif letter.startswith("_"):
                    column += 1
            column = 1
            row += 1
    return cnf, clauses, unitClauses, int(sudokuSize)

def read_the_solution(file, gridSize):
    f = open(file, "r")
    f1 = f.readlines()
    solutionFromFile = ""
    for x in f1:
        if x.startswith("v "):
            solutionFromFile += x
    solutionFromFile = solutionFromFile.split(" ")
    # print(len(solution), solution)

    solution = []
    for s in solutionFromFile:
        if any(c.isalpha() for c in s):
            s = s.split("v")
            if s[0].isdigit:
                solution.append(s[0].rstrip())
        else:
            if s.isdigit:
                solution.append(s)

    solution = [int(s) for s in solution if s.isdigit() and int(s) > (gridSize * gridSize + gridSize) and int(s) % (gridSize * gridSize) > gridSize - 1 and int(s) % gridSize != 0]
    
    matrixSolution = [[0] * (gridSize - 1) for i in range(gridSize - 1)]

    rowD = gridSize * gridSize
    for number in solution:
        row = int(number/rowD)
        column = int((number - row * rowD) / gridSize)
        value = number - row * rowD - column * gridSize
        matrixSolution[row - 1][column - 1] = value

    print(len(solution))
    #for row in matrixSolution:
    #    print('  '.join([str(elem) for elem in row]))

    return matrixSolution

def writeSolution(readFile, outputFile, matrixSolution, blockSize):
    f = open(readFile,"r")
    f1 = f.readlines()
    w = open(outputFile, "w+")
    
    row = 1
    column = 1
    spaces = len(str(blockSize * blockSize))

    for x in f1:
        if x.startswith("|"):
            text = ""
            a = 0
            x = x.split(" ")
            for letter in x:
                if "|" in letter:
                    a += 1
                if letter.isnumeric():
                    text += letter.rjust(spaces + 1)
                    column += 1
                elif letter.startswith("_"):
                    text += str(matrixSolution[row - 1][column - 1]).rjust(spaces + 1)
                    column += 1
                elif column == 1:
                    text += letter
                elif a == blockSize + 1:
                    text += letter.rjust(3)
                elif "|" in letter:
                    text += letter.rjust(2)
            w.write(text)
            column = 1
            row += 1
            a = 0
        else:
            w.write(x)
    w.close()


if __name__ == '__main__':

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    parser = optparse.OptionParser()
    parser.add_option('-f', '--file',
                      action="store", dest="file",
                      help="Choose Path to file", default='instances/table16-3.txt')

    options, args = parser.parse_args()
    read_file = options.file

    cnf2, clauses2, unitClauses, gridSize = read_a_file(read_file)
    
    blockSize = int(math.sqrt(gridSize))
    sudo_size = gridSize + 1
    clauses1, cnf1, unitClauses = test(gridSize, blockSize, unitClauses)
    clauses1, cnf1, unitClauses = test(gridSize, blockSize, unitClauses)

    cnf3 = ""
    for x in unitClauses:
        cnf3 += str(int(x)) + " 0\n"

    file = re.sub('\.txt$', '', read_file) +".cnf"
    print(file)
    f = open(file, "w+")
    f.write("p cnf " + str(gridSize * sudo_size * sudo_size + gridSize * sudo_size + gridSize) + " " + str(clauses1 + len(unitClauses)) + "\n" + cnf3 + cnf1)
    f.close()
    print(clauses1 + len(unitClauses))

    command = "mkdir -p solution"
    os.system(command)
    output = re.sub('\.cnf$', '', file) +".txt"
    output = output.split("/")
    nameFile = str(output[1])
    output = "solution/" + nameFile
    print(output)
    command = "clasp " + file + " > " + output
    os.system(command)

    matrixSolution = read_the_solution(output, gridSize + 1)
    outputFile = re.sub('\.txt$', '', nameFile) +"-output.txt"
    outputFile = "solution/" +  outputFile
    #command2 = "cp instances/" + nameFile + " " +  outputFile
    #os.system(command2)
    writeSolution(read_file, outputFile, matrixSolution, 6)


