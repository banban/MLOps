#!/bin/sh
# `/sbin/setuser memcache` runs the given command as the user `memcache`.
# If you omit that part, the command will be run as root.
# exec /sbin/setuser memcache /usr/bin/memcached >>/var/log/memcached.log 2>&1
#  ./run: 5: exec: cd: not found
#exec cd /usr/src/app && nohup python3 -m swagger_server &
su - rapidxai
cd /usr/src/app && nohup python3.9 -m swagger_server &