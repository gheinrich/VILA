# for pyfile in ./*.py; do 
# for pyfile in $(find ./ -iname "*.py" | xargs); do
#     echo $pyfile
#     black --line-length 120 $pyfile
#     isort $pyfile
# done

isort . 
black --line-length 120 .