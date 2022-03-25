# ps axf | grep main.py | grep -v grep | awk '{print "kill -9 " $1}' | sh

sudo lsof -w /dev/accel0 | grep main.py | awk '{print "kill -9 " $2}'
sudo lsof -w /dev/accel0 | grep main.py | awk '{print "kill -9 " $2}' | sh