demo1.xlsx:
	emon -collect-edp -f emon.dat -w /usr/bin/taskset 0x7fe /home/taosy/repo/shaoyu/ai-opt-demo/build/demo1/demo1 -t "cpu" -p 20 -N 10
