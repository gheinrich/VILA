# /bin/bash
export SLURM_ACCOUNT=nvr_elm_llm

VILA_CI_MAILIST="ligengz@nvidia.com,jasonlu@nvidia.com,yunhaof@nvidia.com,fuzhaox@nvidia.com,yukangc@nvidia.com,huqinghao@gmail.com"

SECONDS=0
while true; do

mkdir -p dev

# skip if no new commits
git pull
githash=$(git log --format="%H" -n 1)
if [[ $(< dev/githash.txt) == "$githash" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M') No updates since last CI. Skip $githash"
    sleep 60
    continue
else
    echo "$(date '+%Y-%m-%d %H:%M') New updates detected. Start CI"
    echo $githash > dev/githash.txt
fi

# running CIs jobs
which python
bash CIs/continual_local.sh $VILA_CI_MAILIST

# launch CI every 3 hour if new commits
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
