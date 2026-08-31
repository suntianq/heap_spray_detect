#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * net_busy normal-control workload: churns UDP + AF_UNIX loopback traffic so the
 * trace shows sustained sk_buff / sock / inode allocations in the network size
 * buckets at a natural cadence (a few bursts/sec). A total-op cap keeps the run
 * under the host ring buffer so it does not overrun-invalidate (same pattern as
 * workload_msg / workload_key).
 */

#define MAX_OPS 40000
#define LOOPBACK "127.0.0.1"
#define BUFSZ 8192

static long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);
    if (duration < 1) duration = 1;

    char buf[BUFSZ];
    memset(buf, 'N', sizeof(buf));

    long start = now_ms();
    int ops = 0;
    long prev = start;

    while (now_ms() - start < duration * 1000L && ops < MAX_OPS) {
        /* UDP loopback: bound socket sends to itself; loopback echoes back. */
        int fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd >= 0) {
            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(40000 + (getpid() % 2000));
            addr.sin_addr.s_addr = inet_addr(LOOPBACK);
            if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
                for (int i = 0; i < 40 && ops < MAX_OPS; i++) {
                    sendto(fd, buf, 512 + (i % 8) * 256, 0,
                           (struct sockaddr *)&addr, sizeof(addr));
                    recvfrom(fd, buf, sizeof(buf), MSG_DONTWAIT, NULL, NULL);
                    ops++;
                }
            }
            close(fd);
        }

        /* AF_UNIX socketpair: dense bidirectional churn. */
        int sp[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0) {
            for (int i = 0; i < 60 && ops < MAX_OPS; i++) {
                write(sp[0], buf, 1024 + (i % 4) * 512);
                read(sp[1], buf, sizeof(buf));
                ops++;
            }
            close(sp[0]);
            close(sp[1]);
        }

        /* keep ~5 bursts/sec: no stall on the trace buffer. */
        long elapsed = now_ms() - prev;
        if (elapsed < 200) usleep((200 - elapsed) * 1000);
        prev = now_ms();
    }

    fprintf(stderr, "net workload done: %d ops in %ldms\n", ops, now_ms() - start);
    return 0;
}
