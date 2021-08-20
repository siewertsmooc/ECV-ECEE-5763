/*
Function: RT Queueing signal, shared buffer, 
multitasked demonstration

POSIX features used: queueing signals and semaphores

Sam Siewert - 7/8/97
Compiled and tested for: Solaris 2.5

Re-tested on Linux 2.6
*/

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <semaphore.h>
#include <errno.h>
#include <pthread.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
  
pthread_t idle_thr_id;
static int idle_tid;

static char *shared_buffer; /* pointer for 
Unix shared memory */

#define ERROR -1

#define NUMSIGS 10
#define TSIGRTMIN  SIGRTMIN+1
static int sigcount = 0;

static sem_t *sbsem;
static int exit_thread = 0;

void got_user_signal(int signo, siginfo_t *info, void *ignored)
{
  void *ptr_val = info->si_value.sival_ptr;
  int int_val = info->si_value.sival_int;

  sigcount++;

  printf("Caught user signal %d %d times with val=%d\n", 
         signo, sigcount, int_val);

  sleep(1);

  if(sigcount == NUMSIGS)
    exit_thread = 1;

  return;
}


void *idle (void *arg) {

  struct sigaction sa;

  sa.sa_sigaction = got_user_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;

  sigcount = 0;

  if(sigaction(TSIGRTMIN, &sa, NULL)) {
    perror("sigaction");
    exit(1);
  }  

  while(!exit_thread) {

    /* this thread will block all user level threads */
    sleep(1);

    /* CRITICAL SECTION -- read and write of shared buffer */
    if(sem_wait(sbsem) == ERROR)
      perror("sem_wait");
    else {
      printf("*****CS Shared buffer = %s\n", 
shared_buffer);
      strcpy(shared_buffer, "idle task was here!");
    }
    if(sem_post(sbsem) == ERROR)
      perror("sem_post");


    fflush(stdout);

  }

  printf("Child idle thread doing a pthread_exit\n");
  fflush(stdout);
  pthread_exit(NULL);

}


main() {
  key_t shmkey = 41467;
  int shm_flag;
  int sbid;

  int i;
  union sigval val;

  /* Setup POSIX binary semaphore */
  sbsem = (sem_t*)malloc(sizeof(sem_t));
  if(sem_init(sbsem, 1, 1) == ERROR) {
    perror("semaphore create");
  }


  /* Setup Unix shared memory for shared_buffer */
  shm_flag = 0;
  shm_flag |= IPC_CREAT | 0666;

  if((sbid = shmget(shmkey, 256, shm_flag)) < 0)
    perror("shared memory get");

  if((shared_buffer = (char *) shmat(sbid, NULL, 0)) == (char *) -1)
    perror("signal shmat");


  if((idle_tid = fork()) == 0) {

    printf("This is the idle process\n");

    /* since we have a copy of the sbid, just go ahead and attach */
    if((shared_buffer = (char *) shmat(sbid, NULL, 0)) == (char *) -1)
      perror("idle shmat");

    if(pthread_create(&idle_thr_id, NULL, idle, NULL)) {
      perror("creating second idle thread\n");
      exit(-1);
    }
    else {
      /* make process wait on idle thread to do pthread_exit */
      pthread_join(idle_thr_id, NULL);

      printf("Idle thread exited, process exiting\n");
      fflush(stdout);

      exit(0);

    }


  }

  printf ("Signal generator: Idle task/process to signal has been spawned\n");
  fflush(stdout);


  for(i=0;i<NUMSIGS;i++) {

    val.sival_int = i;
    if((sigqueue(idle_tid, TSIGRTMIN, val)) == ERROR)
      printf("Signal queue error\n");
    else
      printf ("Signal %d thrown with val=%d\n", TSIGRTMIN, i);

    /* CRITICAL SECTION -- read and write of shared buffer */
    if(sem_wait(sbsem) == ERROR)
      perror("sem_wait");
    else {
      printf("*****CS Shared buffer = %s\n", shared_buffer);
      strcpy(shared_buffer, "signal task was here!");
    }
    if(sem_post(sbsem) == ERROR)
      perror("sem_post");

    fflush(stdout);

    if(!(i%3))  {
      sleep(4);
    }

  }

  /* parent task will wait for child and its threads to exit */
  waitpid(idle_tid, NULL, 0);

  printf("Multithreaded child just shutdown\n");
  fflush(stdout);

  if(sem_destroy(sbsem) == ERROR)
    perror("sem destroy");

  printf ("\nAll done\n");

  exit(0);
}
