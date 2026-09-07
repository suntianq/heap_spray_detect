#include <arpa/inet.h>
#include <fcntl.h>
#include <keyutils.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/*
 * business normal workload: simulates a small server doing mixed, realistic
 * work -- periodic file I/O, pipes, loopback sockets, process spawn, and
 * occasional low-volume (size-varied) SysV message-queue / keyctl activity.
 *
 * Why not churn like workload_msg / workload_key: those concentrate tens of
 * thousands of same-size allocations on the exact kernel call sites that
 * heap-spray exploits use, so a one-class model trained on them learns the
 * spray signature as "normal" and attacks using that path become invisible
 * (CVE-2017-8824 recall 0%: its spray matched the keyctl churn 1:1 in
 * call_site + size; CVE-2010-2959 6%). Real business traffic touches many
 * subsystems at modest rates with naturally varied sizes, so no single
 * (call_site, size) signature dominates and a spray burst stands out by its
 * rate/burst shape instead of being buried in the normal profile.
 *
 * Volume is deliberately bounded (op cap + paced cycles) so the trace stays
 * under the host ring buffer (same pattern as the other workloads). Runs
 * until the duration expires: a server keeps working, it does not burst and
 * quit after two seconds.
 */

#define MAX_OPS 30000
#define WORKDIR "/var/tmp/heap_business_work"
#define CYCLE_MS 150

static unsigned int rng_state;

static unsigned int rand_u32(void) {
    /* xorshift32: tiny PRNG, reseeded per run with time^pid so runs vary. */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int rand_range(int lo, int hi) {
    return lo + (int)(rand_u32() % (unsigned int)(hi - lo + 1));
}

static long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

static int fs_tick(void) {
    char path[256], rb[4096], data[8192];
    int ops = 0;
    memset(data, 'B', sizeof(data));
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof(path), "%s/f%06u.dat", WORKDIR, rand_u32() & 0xffff);
        int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
        if (fd < 0) continue;
        size_t size = (size_t)rand_range(1, 8) * 1024; /* 1..8KB: spread buckets */
        write(fd, data, size);
        lseek(fd, 0, SEEK_SET);
        while (read(fd, rb, sizeof(rb)) > 0) { }
        close(fd);
        unlink(path);
        ops += 2;
    }
    return ops;
}

static int pipe_tick(void) {
    int fds[2];
    char buf[4096];
    if (pipe(fds) < 0) return 0;
    memset(buf, 'P', sizeof(buf));
    int n = rand_range(2, 6);
    for (int i = 0; i < n; i++) {
        write(fds[1], buf, (size_t)rand_range(1, 8) * 256); /* 256B..2KB */
        read(fds[0], buf, sizeof(buf));
    }
    close(fds[0]);
    close(fds[1]);
    return n;
}

static int net_tick(void) {
    char buf[4096];
    memset(buf, 'N', sizeof(buf));
    int ops = 0;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd >= 0) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(40000 + (getpid() % 2000));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            int n = rand_range(2, 5);
            for (int i = 0; i < n; i++) {
                sendto(fd, buf, 128 + rand_range(0, 15) * 128, 0,
                       (struct sockaddr *)&addr, sizeof(addr));
                recvfrom(fd, buf, sizeof(buf), MSG_DONTWAIT, NULL, NULL);
                ops++;
            }
        }
        close(fd);
    }
    return ops;
}

static int proc_tick(void) {
    int ops = 0;
    int n = rand_range(1, 2);
    for (int i = 0; i < n; i++) {
        pid_t pid = fork();
        if (pid < 0) break;
        if (pid == 0) {
            execl("/bin/true", "true", (char *)NULL);
            _exit(127);
        }
        int status;
        waitpid(pid, &status, 0);
        ops++;
    }
    return ops;
}

/* Occasional, size-varied SysV message-queue usage: a few messages per tick,
 * payload sizes drawn across buckets -- the opposite of a same-size spray. */
static int ipc_tick(void) {
    struct {
        long mtype;
        char mtext[4096];
    } msg;
    int qid = msgget(IPC_PRIVATE, 0666 | IPC_CREAT);
    if (qid < 0) return 0;
    int ops = 0;
    int n = rand_range(4, 12);
    for (int i = 0; i < n; i++) {
        int size = rand_range(64, 4096);
        msg.mtype = 1 + (int)(rand_u32() % 4);
        memset(msg.mtext, 'M', (size_t)size);
        if (msgsnd(qid, &msg, size, IPC_NOWAIT) == 0) ops++;
    }
    while (msgrcv(qid, &msg, sizeof(msg.mtext), 0, IPC_NOWAIT) > 0) ops++;
    msgctl(qid, IPC_RMID, NULL);
    return ops;
}

/* Occasional, size-varied keyctl usage: a few keys per tick at realistic
 * volume (real servers rarely churn keyrings; the exploit spray does). */
static int key_tick(void) {
    char desc[64], payload[1024];
    memset(payload, 'K', sizeof(payload));
    int ops = 0;
    int n = rand_range(2, 6);
    for (int i = 0; i < n; i++) {
        int size = rand_range(32, 1024);
        snprintf(desc, sizeof(desc), "biz_%u", rand_u32() & 0xffff);
        key_serial_t key = add_key("user", desc, payload, size,
                                   KEY_SPEC_PROCESS_KEYRING);
        ops++;
        if (key >= 0 && (i & 1)) {
            keyctl(KEYCTL_REVOKE, key);
            ops++;
        }
    }
    keyctl(KEYCTL_CLEAR, KEY_SPEC_PROCESS_KEYRING);
    return ops + 1;
}

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);
    if (duration < 1) duration = 1;

    rng_state = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    if (rng_state == 0) rng_state = 0xa5a5a5a5u;

    /* Allow a queue to hold a full varied batch (default msgmnb is 16KB). */
    system("sysctl -w kernel.msgmnb=4194304 >/dev/null 2>&1");

    char mk[256];
    snprintf(mk, sizeof(mk), "mkdir -p %s 2>/dev/null", WORKDIR);
    system(mk);

    long start = now_ms();
    long prev = start;
    long cycle = 0;
    int ops = 0;

    while (now_ms() - start < duration * 1000L && ops < MAX_OPS) {
        ops += fs_tick();
        ops += pipe_tick();
        ops += net_tick();
        ops += proc_tick();
        cycle++;
        /* IPC/keyrings exist on a real server, but sparse and size-varied:
         * never a same-size burst concentrated on one call site (spray shape). */
        if (cycle % 8 == 0) ops += ipc_tick();
        if (cycle % 11 == 0) ops += key_tick();

        long elapsed = now_ms() - prev;
        if (elapsed < CYCLE_MS) usleep((CYCLE_MS - elapsed) * 1000);
        prev = now_ms();
    }

    fprintf(stderr, "business workload done: %d ops in %ldms (%ld cycles)\n",
            ops, now_ms() - start, cycle);
    return 0;
}
