#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/*
 * fork_stress normal-control workload: repeatedly forks short-lived children
 * that do a little syscall churn then exit, churning task_struct / mm_struct /
 * pid / kernel_stack allocations. It is a HARD NEGATIVE for the discriminator:
 * high allocation rate and multi-process dominance, but no spray of the target
 * slab. A total-op cap keeps the run under the host ring buffer so it does not
 * overrun-invalidate (same pattern as the other workloads).
 */

#define MAX_FORKS 20000

static long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);
    if (duration < 1) duration = 1;

    long start = now_ms();
    int nfork = 0;
    long prev = start;

    while (now_ms() - start < duration * 1000L && nfork < MAX_FORKS) {
        for (int i = 0; i < 4 && nfork < MAX_FORKS; i++) {
            pid_t pid = fork();
            if (pid == 0) {
                /* child: brief nanosleep (alloc/free churn) then exit. */
                struct timespec ts = { .tv_sec = 0, .tv_nsec = 2000000 };
                nanosleep(&ts, NULL);
                _exit(0);
            } else if (pid > 0) {
                nfork++;
            }
        }
        /* reap completed children so no zombie pile-up. */
        int st;
        while (waitpid(-1, &st, WNOHANG) > 0) { }
        /* keep ~5 bursts/sec: no stall on the trace buffer. */
        long elapsed = now_ms() - prev;
        if (elapsed < 200) usleep((200 - elapsed) * 1000);
        prev = now_ms();
    }

    /* drain remaining children before exiting. */
    int st;
    while (waitpid(-1, &st, 0) > 0) { }
    fprintf(stderr, "fork workload done: %d forks in %ldms\n", nfork, now_ms() - start);
    return 0;
}
