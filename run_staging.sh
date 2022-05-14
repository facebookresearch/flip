# VM_NAME=kmh-tpuvm-v3-128-1
VM_NAME=kmh-tpuvm-v3-256-3
echo $VM_NAME

# ------------------------------------------------
# copy all files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`
export STAGEDIR=/kmh_data/staging/${now}-${salt}-code

echo 'Copying files...'
rsync -a . $STAGEDIR --exclude=tmp
echo 'Done copying.'

chmod 777 $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

for seed in 0
do
source run_remote.sh

echo sleep 1m
sleep 1m
source run_kill_remote.sh
done
