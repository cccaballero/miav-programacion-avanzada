#!/usr/bin/env python
from mpi4py import MPI

import math
import numpy
import sys
import argparse
from multiprocessing import Pool
from functools import partial


def secuential_integrate(a,b,n,func):
    #se calcula el ancho
    h = (b-a)/n
    i = 0.0
    # se evalúa la función para cada uno de los fragmentos y se suman los resultados
    for j in range(n):
        i = i + func(a+j*h)
    # los resultados de la ecaluación de la función se multiplican por h para obtener el valor de la integral
    i = i*h
    return i

def mpi_integrate(a, b, n, func):
    # se obtiene el intracommunicator COMM_WORLD
    comm = MPI.COMM_WORLD
    # se obtiene el proceso actual
    rank = comm.Get_rank()
    # se obtiene el total de procesos
    size = comm.Get_size()

    # se divide n segun size
    n = n//size

    def integral(a, h, n):
        integ = 0.0
        for j in range(n):
            a_j = a + (j+n*rank) * h
            integ += func(a_j)
        return integ

    #se calcula el ancho
    h = (b - a) / (n * size)

    # se calcula la integral para el proceso actual
    my_int = numpy.full(1, integral(a, h, n))

    
    if rank == 0:
        # si es el proceso 0, se dispone a obtener los datos de los otros procesos 
        integral_sum = my_int[0]
        # ciclo en el que se obtienen los datos de los otros procesos y se termina el calculo de la integral
        for p in range(1, size):
            comm.Recv(my_int, source=p)
            integral_sum += my_int[0]
        integral_sum = integral_sum * h

        print(integral_sum)
    else:
        # si no es el proceso 0, evía el resultado al proceso 0
        comm.Send(my_int, dest=0)



_func = None

def multiprocessing_integral(a, h, n, rank):
    integ = 0.0
    for j in range(n):
        a_j = a + (j+n*rank) * h
        integ += _func(a_j)
    return integ


def multiprocessing_integrate(a, b, n, func):
    # la asignación de _func permitirá usar la lambda func con multiprocessing
    global _func
    _func = func

    # size define la cantidad de procesos a ejecutar
    size = 4
    
    # se divide n segun size
    n = n//size
    #se calcula el ancho
    h = (b - a) / (n * size)

    integral_values = []
    # se declara un Pool para 'size' procesos
    with Pool(size) as p:
        # se instancia una funcion parcial para permitir multiples argumentos a multiprocessing 
        func = partial(multiprocessing_integral, a, h, n)
        # se ejecutan los procesos y se espera por los resultados
        integral_values = p.map(func, range(size))
    integral_sum = 0.0
    #se suman los resultados y se multiplica por h
    for integral_value in integral_values:
        integral_sum += integral_value
    integral_sum = integral_sum * h
    return integral_sum

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("function", type=int, default=1, choices=[1,2,3], help='Define la función a usar (1) x**2+math.cos(x) (2) (8*math.e**(1-x))+(7*math.log(x)) (3) 10+x**2-10*math.cos(math.pi*x)**2')

    parser.add_argument("areas", type=int, default=10000, help='Define los n rectangulos a utilizar para el calculo')
    
    parser.add_argument('--avalue', type=float, default=None, help="Valor de a")

    parser.add_argument('--bvalue', type=float, default=None, help="Valor de b")

    parser.add_argument('--secuential', action="store_true", help="Utilizará un algoritmo secuancial para el cálculo")

    parser.add_argument('--mpi', action="store_true", help="Utilizará un algoritmo paralelo ulilizando mpi2py (requiere ser ejecutado con mpiexec)")

    parser.add_argument('--multiprocessing', action="store_true", help="Utilizará un algoritmo paralelo ulilizando multiprocessing")

    args = parser.parse_args()

    functions = {
        1: (
            0.0 if not args.avalue else args.avalue,
            4.0 if not args.bvalue else args.bvalue,
            lambda x: x**2+math.cos(x)
            ),
        2: (
            0.5 if not args.avalue else args.avalue,
            2.5 if not args.bvalue else args.bvalue,
            lambda x: (8*math.e**(1-x))+(7*math.log(x))
            ),
        3: (
            -2.0 if not args.avalue else args.avalue,
            2.0 if not args.bvalue else args.bvalue,
            lambda x: 10+x**2-10*math.cos(math.pi*x)**2
            ),
    }

    data = (
        functions[args.function][0], 
        functions[args.function][1], 
        args.areas,
        functions[args.function][2]
        )
    
    if args.secuential:
        print(secuential_integrate(*data))
    elif args.mpi:
        mpi_integrate(*data)
    elif args.multiprocessing:
        print(multiprocessing_integrate(*data))
    
