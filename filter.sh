#!/bin/bash
# /home/arong/go/bin/s2prot -indent=false $1 | grep '"AssignedRace":"Prot".*PlayerID":1.*"AssignedRace":"Prot".*PlayerID":2.*"Title":"Blueshift LE"' > /dev/null
/home/arong/go/bin/s2prot -indent=false $1 | grep '"AssignedRace":"Prot".*PlayerID":1.*"AssignedRace":"Prot".*PlayerID":2' > /dev/null
# home/arong/go/bin/s2prot -details -indent=false $1 | grep -E '"title":[^}]+' --only-matching
status=$?
if [ $status -eq 0 ]; then
	echo $1
fi
