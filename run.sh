#!/bin/bash


run_calc () {
    echo ''
    echo "Función $1, n=$2"
    echo ''
    echo '---- secuencial -----'
    time python3 main.py $1 $2 --secuential

    echo ''
    echo '---- mpi -----'
    time mpiexec -n 4 python main.py $1 $2 --mpi

    echo ''
    echo '---- multiprocessing -----'
    time python main.py $1 $2 --multiprocessing

    echo '-------------------------------'
}

run_calc 1 10000
run_calc 1 100000
run_calc 1 1000000

run_calc 2 10000
run_calc 2 100000
run_calc 2 1000000

run_calc 3 10000
run_calc 3 100000
run_calc 3 1000000
