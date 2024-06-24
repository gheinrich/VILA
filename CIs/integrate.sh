# /bin/bash
export SLURM_ACCOUNT=nvr_elm_llm


SECONDS=0
while true; do

# WORKDIR=~/workspace/VILA-internal
# cd $WORKDIR
mkdir -p dev

git pull
githash=$(git log --format="%H" -n 1)
# skip if no new commits
if [[ $(< dev/githash.txt) == "$githash" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M') No updates since last CI. Skip $githash"
    sleep 60
    continue
else
    echo "$(date '+%Y-%m-%d %H:%M') New updates detected. Start CI"
    echo $githash > dev/githash.txt
fi

which python
bash CIs/continual_local.sh "ligengz@nvidia.com,jasonlu@nvidia.com,yunhaof@nvidia.com,fuzhaox@nvidia.com"

# every 3 hour if new commits
while true; do
    if [ "$SECONDS" -gt "10800" ]; then
        SECONDS=0
        break
    else
        echo "$(date '+%Y-%m-%d %H:%M') CI waiting progress, $SECONDS of 10800 seconds"
        sleep 60
    fi 
done
done 

# hourly
# 5 * * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> ~/workspace/VILA-internal/dev/crontab.txt

# daily
# 5 */4 * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> ~/workspace/VILA-internal/dev/crontab.txt
