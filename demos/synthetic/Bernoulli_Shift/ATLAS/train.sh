set -ex
expr_root=`pwd`


while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -o|--output)
            expr_root=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            exit 1
            ;;
    esac
done

output=${expr_root}/output
mkdir -p $output
date > $output/log.txt

while [ ! -f "main.py" ]
do
    cd ..
done


python -u main.py \
    --expr_space ${expr_root} | tee -a $output/log.txt

