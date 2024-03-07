cd ~/workspace/VILA-internal
mkdir -p dev
> ~/workspace/VILA-internal/dev/crontab.txt
git pull

source activate vila

which python
bash CIs/continual_local.sh


"""
# hourly
5 * * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> /home/ligengz/workspace/VILA/dev/

# daily
5 0 * * * bash ~/workspace/VILA-internal/CIs/integrate.sh >> /home/ligengz/workspace/VILA/dev/
"""