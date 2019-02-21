s2prot -indent=false $1 | grep "\"Prot\".*PlayerID\":1" > /dev/null
status=$?
if [ $status -eq 0 ]; then
	echo $1
fi
