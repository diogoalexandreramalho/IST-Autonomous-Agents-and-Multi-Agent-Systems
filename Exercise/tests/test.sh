#!/bin/bash -e

# Assumptions:
#   - This test runs in a Unix machine, with:
#       + bash
#       + coreutils
#       + xargs (findutils)
#   - The test files:
#       + Are contained in a subdirectory, by default "students-tests"
#       + Each test as exactly 2 files:
#           * XXXinput.txt, the input
#           * XXXoutput.txt, the expected and correct output
#       + There is only one right answer, and it should exactly match the output



# Here is the command to run 1 instance of the project, with stdin as input and
# stdout as output. If this setup is not possible, you will need to change this
# script heavily.
# Also, if the program as any error, the exit status must be != 0
run() {
./exercise
}


single_job_cmp() {
  run <"$1" >"${1%input.txt}output.txt_hyp" 2>"${1%input.txt}err.log"
  diff -wq "${1%input.txt}output.txt" "${1%input.txt}output.txt_hyp">/dev/null
  res="$?"
  if [ "$res" -eq 0 ]; then rm "${1%input.txt}output.txt_hyp" ; fi
  if [ -s "${1%input.txt}err.log" ] ; then res=2 ;
                            else rm "${1%input.txt}err.log" ; fi
  printf "%q\0%q\n" "$res" "$1"
}

single_job_diff() {
  run < "$1" | diff "${1%input.txt}output.txt" - | less
}

shell_aggregator() {

  local passed=0
  local failed=0
  local error=0
  local total=0
  local ret=0

   while read line ; do
      ret=`cut -b1 $line`

      # increase counters
      total=$((total+1))
      if   [ $ret -eq 0 ]; then passed=$((passed+1));
      elif [ $ret -eq 1 ]; then failed=$((failed+1));
      elif [ $ret -eq 2 ]; then error=$((error+1));
      fi

  done

  printf '\nTotal: %b\nPassed: %b\nFailed: %b\nError in test: %b\n' $total $passed $failed $error

}

awk_aggregator() {

awk 'BEGIN{
         FS="\0"
     }
     {  total++;
        if($1 == 0) {
            passed++;
        } else {
            if($1 == 1) {
                files_wrong[failed++] = $2;
            } else {
                files_error[error++] = $2;
            }
       }
     }
     END{
          if(failed > 0) {
            wrong_files = asort(files_wrong);
            printf "Failed tests:\n";
            for(i=1; i<=wrong_files; ++i) print files_wrong[i];
          }
          if(error > 0) {
            error_files = asort(files_error)
            printf "Errored tests:\n"
            for(i=1; i<=error_files; ++i) print files_error[i];
          }
            printf "\nTotal: %d\nPassed: %d\nFailed: %d\nError in test: %d\n",
            total, passed, failed, error;
        }'
}


# Inspired/taught by
# https://adamdrake.com/command-line-tools-can-be-235x-faster-than-your-hadoop-cluster.html

# To get sequential execution, use N=1
pipeline() {
  find "$TESTS_DIR" -type f -name '*input.txt' -print0 | xargs -0 -I {} -n1 -P"$N" bash -c $TEST' {}' 'tests'  | $AGGREGATOR
}





# Number of processes to run (by default)
N=8
# Default test
TEST=single_job_cmp
# Default aggregator
AGGREGATOR=awk_aggregator

TESTS_DIR="."

export -f $TEST run
export -f $AGGREGATOR
pipeline;

