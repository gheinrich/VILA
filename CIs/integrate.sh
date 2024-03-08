SECONDS=0
while true; do

WORKDIR=~/workspace/VILA-internal
cd $WORKDIR
mkdir -p dev
> $WORKDIR/dev/crontab.txt
git pull

source activate vila

which python
bash CIs/continual_local.sh


while true; do
    if [ "$SECONDS" -gt "14400"]; then
        SECONDS=0
        break
    else
        sleep 10
    fi 
done


done 

"""
# hourly
5 * * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> ~/workspace/VILA-internal/dev/crontab.txt

# daily
5 */4 * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> ~/workspace/VILA-internal/dev/crontab.txt
"""