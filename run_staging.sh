# VM_NAME=kmh-tpuvm-v3-128-1
VM_NAME=kmh-tpuvm-v3-256-4
echo $VM_NAME

# ------------------------------------------------
# copy all files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`
export STAGEDIR=/kmh_data/staging/${now}-${salt}-code

sudo chmod 777 $STAGEDIR

echo 'Copying files...'
rsync -a . $STAGEDIR --exclude=tmp
echo 'Done copying.'

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

for seed in 0 1 2 3
do
source run_remote.sh

echo sleep 1m
sleep 1m
source run_kill_remote.sh
done