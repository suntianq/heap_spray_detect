#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>
#include <time.h>

/*
 * msg_msg normal-control workload: churns System V message queues so the trace
 * shows sustained kmalloc activity in the msg_msg size buckets (256/2048).
 *
 * The default msgmnb (16 KB) would cap a queue at ~64x256B or 8x2048B messages,
 * after which a blocking msgsnd stalls forever and the run times out. We send
 * with IPC_NOWAIT, drain the queue, remove it, and re-create it. A total-op cap
 * keeps the trace under the host ring buffer (no overrun-invalidating runs).
 */

#define MSG_PER_CYCLE 2000
#define MAX_OPS 60000
#define DEFAULT_MSG_SIZE 256

struct msgbuf {
    long mtype;
    char mtext[8192];
};

int main(int argc, char *argv[]) {
    int duration = 30;
    int msg_size = DEFAULT_MSG_SIZE;
    if (argc > 1) duration = atoi(argv[1]);
    if (argc > 2) msg_size = atoi(argv[2]);
    if (msg_size < 1 || msg_size > (int)sizeof(((struct msgbuf *)0)->mtext))
        msg_size = DEFAULT_MSG_SIZE;

    /* Allow a single queue to hold a dense batch for both msg sizes. */
    system("sysctl -w kernel.msgmnb=4194304 >/dev/null 2>&1");

    struct msgbuf *msg = malloc(sizeof(long) + msg_size);
    if (!msg) return 1;
    msg->mtype = 1;
    memset(msg->mtext, 'A', msg_size);

    time_t start = time(NULL);
    int total_ops = 0;

    while (time(NULL) - start < duration && total_ops < MAX_OPS) {
        int qid = msgget(IPC_PRIVATE, 0666 | IPC_CREAT);
        if (qid < 0) continue;
        for (int i = 0; i < MSG_PER_CYCLE; i++) {
            if (msgsnd(qid, msg, msg_size, IPC_NOWAIT) < 0) break;
            total_ops++;
        }
        while (msgrcv(qid, msg, msg_size, 0, IPC_NOWAIT) > 0)
            total_ops++;
        msgctl(qid, IPC_RMID, NULL);
        usleep(3000);
    }

    fprintf(stderr, "msg_msg workload done: %d ops in %lds (msg_size=%d)\n",
            total_ops, (long)(time(NULL) - start), msg_size);
    free(msg);
    return 0;
}
