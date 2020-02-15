#!/bin/bash


run_calc () {

    secuential="python3 main.py $1 $2 --secuential > /dev/null"
    mpi="mpiexec -n 4 python main.py $1 $2 --mpi > /dev/null"
    multiprocessing="python main.py $1 $2 --multiprocessing > /dev/null"

    myString=$(printf "%10s")

    echo ''
    echo "Función $1, n=$2"
    echo ''
    echo '---- secuencial -----'
    /usr/bin/time -f "mem=%K RSS=%M time=%E cpu.sys=%S" /bin/sh -c "${myString// /$secuential;}"

    echo ''
    echo '---- mpi -----'
    /usr/bin/time -f "mem=%K RSS=%M time=%E cpu.sys=%S" /bin/sh -c "${myString// /$mpi;}"

    echo ''
    echo '---- multiprocessing -----'
    /usr/bin/time -f "mem=%K RSS=%M time=%E cpu.sys=%S" /bin/sh -c "${myString// /$multiprocessing;}"

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
