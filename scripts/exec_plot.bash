for file in `\find data -name '*.csv'`; do
    before_period=${file%%.*}
    echo "$before_period"
    fig_name=${before_period}.png
    title=${before_period#*/}
    python3 progress_to_plot.py -d ${file} -f ${fig_name} -t ${title}
done
