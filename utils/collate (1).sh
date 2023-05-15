for script in */*.*
do
    # echo "Found script: $script"

    filename=$(basename ${script})
    #echo "Found file: $filename"

    parentdir="$(dirname "$script")"
    #echo "Found dir: $parentdir"

    mv $script "$parentdir$filename"

done
