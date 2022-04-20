for file in `\find data -name '*.csv'`; do
    echo "$file"
    fig_name=${file%%.*}.png
    python progress_to_plot.py -d ${file} -f ${fig_name}
done
