#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/*
 * fs_io normal-control workload: churns create/write/fsync/read/unlink on a
 * disk-backed directory so the trace shows sustained inode / dentry / buffer_head
 * / page-cache allocation activity distinct from the IPC and key workloads.
 *
 * The guest runs with -snapshot, so writes are ephemeral and never pollute the
 * image. Default workdir is /var/tmp (on the root ext4 fs, unlike /tmp which is
 * tmpfs); override with HEAP_FS_DIR. A total-op cap keeps the run under the host
 * ring buffer (no overrun-invalidating runs).
 */

#define MAX_OPS 20000
#define WORKDIR "/var/tmp/heap_fs_work"
#define BUFSZ (512 * 1024)

static long now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + tv.tv_usec / 1000L;
}

int main(int argc, char *argv[]) {
    int duration = 30;
    if (argc > 1) duration = atoi(argv[1]);
    if (duration < 1) duration = 1;

    const char *dir = getenv("HEAP_FS_DIR");
    if (!dir) dir = WORKDIR;
    char path[1024];

    char *buf = malloc(BUFSZ);
    if (!buf) return 1;
    memset(buf, 'F', BUFSZ);

    char mk[1024];
    snprintf(mk, sizeof(mk), "mkdir -p %s 2>/dev/null", dir);
    system(mk);

    long start = now_ms();
    int ops = 0;
    long prev = start;

    while (now_ms() - start < duration * 1000L && ops < MAX_OPS) {
        for (int i = 0; i < 3 && ops < MAX_OPS; i++) {
            snprintf(path, sizeof(path), "%s/f%06d_%d", dir, getpid(), i);
            int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
            if (fd < 0) continue;
            /* alternate small / medium / large so several bucket sizes show up. */
            size_t size = (i % 3 == 0) ? 4096 : (i % 3 == 1) ? 65536 : BUFSZ;
            write(fd, buf, size);
            fsync(fd);
            lseek(fd, 0, SEEK_SET);
            char rb[4096];
            while (read(fd, rb, sizeof(rb)) > 0) { }
            close(fd);
            unlink(path);
            ops++;
        }
        /* keep ~4 bursts/sec: no stall on the trace buffer. */
        long elapsed = now_ms() - prev;
        if (elapsed < 250) usleep((250 - elapsed) * 1000);
        prev = now_ms();
    }

    fprintf(stderr, "fs workload done: %d ops in %ldms\n", ops, now_ms() - start);
    free(buf);
    return 0;
}
