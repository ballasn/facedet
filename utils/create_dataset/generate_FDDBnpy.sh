#! /bin/sh




EXE=/u/ballasn/project/facedet/facedet/utils/create_dataset/lists_to_npy.py
DATA=/u/ballasn/project/facedet/facedet/utils/create_dataset/references/FDDB/fold
OUT=/data/lisatmp3/ballasn/facedet/datasets/FDDB/fold




for i in `seq 1 10`; do
    echo "Processed Fold $i"

    tmpfile=`mktemp`
    echo $tmpfile

    for j in `seq 1 10`; do
        if [ $i -eq $j ]; then
            continue
        fi
        echo $i $j
        cat "${DATA}${j}.txt" >> $tmpfile
        #echo $i $j
    done

    python $EXE $tmpfile 16 ${OUT}${i}.npy
    rm $tmpfile
done
